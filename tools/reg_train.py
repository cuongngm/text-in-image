import os
import shutil
import argparse
import random
import yaml
import numpy as np
import distance
import torch
import torch.nn.functional as f
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from ultocr.loader.recognition.reg_loader import DistCollateFn
from ultocr.loader.recognition.translate import LabelTransformer
from ultocr.model.recognition.master import greedy_decode_with_probability
from ultocr.utils.reg_metrics import AverageMetricTracker
from ultocr.utils.utils_function import create_module
from ultocr.utils.logger import TrainLog


def fix_random_seed_for_reproduce(seed):
    # fix random seeds for reproducibility,
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for current gpu
    torch.cuda.manual_seed_all(seed)  # for all gpu
    torch.backends.cudnn.benchmark = False  # if benchmark=True, speed up training, and deterministic will set be False
    torch.backends.cudnn.deterministic = True  # which can slow down training considerably


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, lr_scheduler, max_len_step, cfg, logger):
        self.config = cfg
        self.distributed = cfg['trainer']['distributed']
        if self.distributed:
            self.local_master = (cfg['trainer']['local_rank'] == 0)
            self.global_master = (dist.get_rank() == 0)
        else:
            self.local_master = True
            self.global_master = True
        self.logger = logger
        self.device, self.device_ids = self.prepare_device(cfg['trainer']['local_rank'],
                                                           cfg['trainer']['local_world_size'])
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.epochs = cfg['trainer']['epochs']
        self.checkpoint_dir = cfg['trainer']['save_dir']

        self.start_epoch = 1
        self.resume = cfg['trainer']['resume']

        self.monitor_best = np.inf

        if self.resume:
            self.resume_checkpoint(cfg['trainer']['resume_path'])
        if self.distributed:
            self.model = DDP(self.model, device_ids=self.device_ids, output_device=self.device_ids[0],
                             find_unused_parameters=True)
        self.len_step = max_len_step
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.do_validation = (self.val_loader is not None and cfg['trainer']['do_validation'])
        self.validation_epoch = cfg['trainer']['validation_epoch']
        self.lr_scheduler = lr_scheduler

        log_step = cfg['trainer']['log_step_interval']
        self.log_step = log_step
        self.val_step_interval = cfg['trainer']['val_step_interval']
        self.early_stop = cfg['trainer']['early_stop']

        self.train_metrics = AverageMetricTracker('loss')
        self.val_metrics = AverageMetricTracker('loss', 'word_acc', 'word_case_insensitive',
                                                'edit_distance_acc')

    def train(self):
        if self.distributed:
            dist.barrier()  # syncing machines before training
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            torch.cuda.empty_cache()
            result_dict = self.train_epoch(epoch)
            if self.do_validation and epoch % self.validation_epoch == 0:
                val_metric_res_dict = self.valid_epoch()
                val_res = f"\nValidation result after {epoch} epoch:" \
                          f"Word_acc: {val_metric_res_dict['word_acc']:.6f}" \
                          f"Word_acc_case_ins: {val_metric_res_dict['word_acc_case_ins']:.6f}" \
                          f"Edit_distance_acc: {val_metric_res_dict['edit_distance_acc']:.6f}"
            else:
                val_res = ''
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.logger.info('[Epoch end] Epoch:[{}/{}] Loss: {:.6f} LR: {:.8f}'
                             .format(epoch, self.epochs, result_dict['loss'], self.get_lr()) + val_res)
            best = False
            if self.do_validation and epoch % self.validation_epoch:
                best, not_improved_count = self.is_best_monitor_metric(best, not_improved_count, val_metric_res_dict)
                if not_improved_count > self.early_stop:
                    self.logger.info('Validation performance didn\'t improve for {} epochs.'
                                     'Training stops'.format(self.early_stop))
                    break
                if best:
                    self.save_checkpoint(epoch, save_best=best)

    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for step_idx, input_data_item in enumerate(self.train_loader):
            batch_size = input_data_item['batch_size']
            images = input_data_item['images']
            text_label = input_data_item['labels']
            step_idx += 1
            images = images.to(self.device)
            target = LabelTransformer.encode(text_label)
            target = target.to(self.device)
            target = target.permute(1, 0)
            with torch.autograd.set_detect_anomaly(True):
                outputs = self.model(images, target[:, :-1])
                loss = f.cross_entropy(outputs.contiguous().view(-1, outputs.shape[-1]),
                                       target[:, 1:].contigous().view(-1), ignore_index=LabelTransformer.PAD)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_total = images.shape[0]
            reduced_loss = loss.item()
            if self.distributed:
                reduced_metrics_tensor = torch.tensor([batch_total, reduced_loss]).float().to(self.device)
                batch_total, reduced_loss = reduced_metrics_tensor.cpu().numpy()
                reduced_loss = reduced_loss / dist.get_world_size()
            global_step = (epoch - 1) * self.len_step + step_idx - 1
            self.train_metrics.update('loss', reduced_loss, batch_total)
            if step_idx % self.log_step == 0 or step_idx == 1:
                self.logger.info('Train Epoch:[{}/{}] Step:[{}/{}] Loss:{:.6f} Loss avg:{:.6f} LR:{:.8f}'
                                 .format(epoch, self.epochs, step_idx, self.len_step,
                                         self.train_metrics.val('loss'),
                                         self.train_metrics.avg('loss'),
                                         self.get_lr()))
            if step_idx == self.len_step:
                break
        log_dict = self.train_metrics.result()
        return log_dict

    def valid_epoch(self, epoch):
        self.model.eval()
        self.val_metrics.reset()
        for step_idx, input_data_item in enumerate(self.val_loader):
            batch_size = input_data_item['batch_size']
            images = input_data_item['images']
            text_label = input_data_item['labels']
            if self.distributed:
                word_acc, word_acc_case_ins, edit_distance_acc, total_distance_ref, batch_total =\
                    self.distributed_predict(batch_size, images, text_label)
            else:
                with torch.no_grad():
                    images = images.to(self.device)
                    if hasattr(self.model, 'module'):
                        model = self.model.module
                    else:
                        model = self.model
                outputs, _ = greedy_decode_with_probability(model, images, LabelTransformer.max_length,
                                                            LabelTransformer.SOS, LabelTransformer.EOS,
                                                            LabelTransformer.PAD, images.device, is_padding=True)
                correct = 0
                correct_case_ins = 0
                total_distance_ref = 0
                total_edit_distance = 0
                for index, (pred, text_gold) in enumerate(zip(outputs[:, 1:], text_label)):
                    predict_text = ''
                    for i in range(len(pred)):
                        if pred[i] == LabelTransformer.EOS: break
                        if pred[i] == LabelTransformer.UNK: continue
                        decoded_char = LabelTransformer.decode(pred[i])
                        predict_text += decoded_char
                    ref = len(text_gold)
                    edit_distance = distance.levenshtein(text_gold, predict_text)
                    total_distance_ref += ref
                    total_edit_distance += edit_distance
                    if predict_text == text_gold:
                        correct += 1
                    if predict_text.lower() == text_gold.lower():
                        correct_case_ins += 1
                batch_total = images.shape[0]
                word_acc = correct / batch_total
                word_acc_case_ins = correct_case_ins / batch_total
                edit_distance_acc = 1 - total_edit_distance/total_distance_ref
            self.val_metrics.update('word_acc', word_acc, batch_total)
            self.val_metrics.update('word_acc_case_insensitive', word_acc_case_ins, batch_total)
            self.val_metrics.update('edit_distance_acc', edit_distance_acc, total_distance_ref)
        val_metric_res_dict = self.val_metrics.result()
        self.model.train()
        return val_metric_res_dict

    def distributed_predict(self, batch_size, images, text_label):
        correct = correct_case_ins = valid_batches = total_edit_distance = total_distance_ref = 0
        if batch_size:
            with torch.no_grad():
                images = images.to(self.device)
                if hasattr(self.model, 'module'):
                    model = self.model.module
                else:
                    model = self.model
                outputs, _ = greedy_decode_with_probability(model, images, LabelTransformer.max_length,
                                                            LabelTransformer.SOS, LabelTransformer.EOS,
                                                            LabelTransformer.PAD, images.device, is_padding=True)
                for idx, (pred, text_gold) in enumerate(zip(outputs[:, 1:], text_label)):
                    predict_text = ''
                    for i in range(len(pred)):
                        if pred[i] == LabelTransformer.EOS: break
                        if pred[i] == LabelTransformer.UNK: continue
                        decoded_char = LabelTransformer.decode(pred[i])
                        predict_text += decoded_char
                    ref = len(text_gold)
                    edit_distance = distance.levenshtein(text_gold, predict_text)
                    total_distance_ref += ref
                    total_edit_distance += edit_distance
                    if predict_text == text_gold:
                        correct += 1
                    if predict_text.lower() == text_gold.lower():
                        correct_case_ins += 1
            valid_batches = 1
        sum_metrics_tensor = torch.tensor([batch_size, valid_batches, correct, correct_case_ins,
                                           total_edit_distance, total_distance_ref]).float().to(self.device)
        sum_metrics_tensor = self.sum_tensor(sum_metrics_tensor)
        sum_metrics_tensor = sum_metrics_tensor.cpu().numpy()
        batch_total, valid_batches = sum_metrics_tensor[:2]
        correct, correct_case_ins, total_edit_distance, total_distance_ref = sum_metrics_tensor[2:]
        word_acc = correct / batch_total
        word_acc_case_ins = correct_case_ins / batch_total
        edit_distance_acc = 1 - total_edit_distance / total_distance_ref
        return word_acc, word_acc_case_ins, edit_distance_acc, total_distance_ref, batch_total

    def sum_tensor(self, tensor: torch.Tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def is_best_monitor_metric(self, best, not_improved_count, val_result_dict, update_not_improved_count=False):
        val_monitor_metric_res = val_result_dict['word_acc']
        improved = val_monitor_metric_res >= self.monitor_best
        if improved:
            self.monitor_best = val_monitor_metric_res
            not_improved_count = 0
            best = True
        else:
            if update_not_improved_count:
                not_improved_count += 1
        return best, not_improved_count

    def save_checkpoint(self, epoch, save_best=False, step_idx=None):
        if hasattr(self.model, 'module'):
            arch_name = type(self.model.module).__name__
            model_state_dict = self.model.module.state_dict()
        else:
            arch_name = type(self.model).__name__
            model_state_dict = self.model.state_dict()
        state = {
            'arch': arch_name,
            'epoch': epoch,
            'model_state_dict': model_state_dict
        }
        if step_idx is None:
            filename = str(self.checkpoint_dir / 'epoch{}.pth'.format(epoch))
        else:
            filename = str(self.checkpoint_dir / 'epoch{}-step{}.pth'.format(epoch, step_idx))
        torch.save(state, filename)
        self.logger.info('Saving checkpoint: {}...'.format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            shutil.copyfile(filename, best_path)
            self.logger.info('Saving current best at epoch {}'.format(epoch))

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group['lr']

    def resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info('Loading checkpoint: {}...'.format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info('Checkpoint loaded. Resume from epoch {}'.format(self.start_epoch))

    def prepare_device(self, local_rank, local_world_size):
        if self.distributed:
            ngpu_per_process = torch.cuda.device_count() // local_world_size
            device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))
            if torch.cuda.is_available() and local_rank != -1:
                torch.cuda.set_device(device_ids[0])
                device = 'cuda'
                self.logger.info('[Process {}] world size = {}, rank = {}, n_gpu/process = {}, device_ids = {}'
                                 .format(os.getpid(), dist.get_world_size(),
                                         dist.get_rank(), ngpu_per_process, device_ids))
            else:
                self.logger.info('Training will be using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, device_ids
        else:
            n_gpu = torch.cuda.device_count()
            n_gpu_use = local_world_size
            if n_gpu_use > 0 and n_gpu == 0:
                self.logger.info('Warning: There is no GPU available on this machine,'
                                 'training will be performed on CPU')
                n_gpu_use = 0
            if n_gpu_use > n_gpu:
                self.logger.info('Warning: The number of GPU configured to use is {}, but only {} are'
                                 'available on this machine'.format(n_gpu_use, n_gpu))
                n_gpu_use = n_gpu
            list_ids = list(range(n_gpu_use))
            if n_gpu_use > 0:
                torch.cuda.set_device(list_ids[0])
                self.logger.info('Training is using GPU {}'.format(list_ids[0]))
                device = 'cuda'
            else:
                self.logger.warning('Training is using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, list_ids


def main(cfg, logger):
    train_dataset = create_module(cfg['functional']['load_data'])(cfg, cfg['dataset']['img_root'],
                                                                  cfg['dataset']['train_txt_file'],
                                                                  training=True)
    val_dataset = create_module(cfg['functional']['load_data'])(cfg, cfg['dataset']['img_root'],
                                                                cfg['dataset']['val_txt_file'],
                                                                training=True)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=cfg['train_loader']['batch_size'],
                              collate_fn=DistCollateFn(training=True),
                              num_workers=cfg['train_loader']['num_workers'],
                              shuffle=True)
    val_loader = DataLoader(val_dataset, sampler=val_sampler,
                            batch_size=cfg['val_loader']['batch_size'],
                            collate_fn=DistCollateFn(training=True),
                            num_workers=cfg['val_loader']['num_workers'])

    logger.info('Dataloader instances have finished. Train datasets: {}, val datasets: {}'.format(len(train_dataset),
                                                                                                  len(val_dataset)))
    max_len_step = len(train_loader)
    model = create_module(cfg['functional']['master'])(cfg)
    logger.info('Model created, trainable parameters: {}'.format(model.model.model_parameters()))
    optimizer = create_module(cfg['optimizer']['functional'])(cfg, model.parameters())
    if cfg['lr_scheduler']['type'] is not None:
        lr_scheduler = create_module(cfg['lr_scheduler']['functional'])(cfg)
    else:
        lr_scheduler = None
    logger.info('Optimizer and lr_scheduler created')
    logger.info('Max_epochs: {}, log_step_interval: {}, Validation_step_interval: {}'
                .format(cfg['trainer']['epochs'],
                        cfg['trainer']['log_step_interval'],
                        cfg['trainer']['val_step_interval']))
    logger.info('Training start...')
    trainer = Trainer(model, optimizer, train_loader, val_loader, lr_scheduler, max_len_step, cfg, logger)
    trainer.train()
    logger.info('Distributed training end...')


def entry_point(cfg):
    # number of process per node
    local_world_size = cfg['trainer']['local_world_size']
    if cfg['trainer']['distributed']:
        if torch.cuda.is_available():
            if torch.cuda.device_count() < local_world_size:
                raise RuntimeError('The number of GPU {} is less than the number of processes {} running on each node'
                                   .format(torch.cuda.device_count(), local_world_size))
        else:
            raise RuntimeError('CUDA is not available, distributed training is not supported')
    else:
        if local_world_size != 1:
            raise RuntimeError('local_world_size must set be to 1, if distributed is set to false')
    logger = TrainLog(LOG_FILE='saved/log/log.txt')
    if cfg['trainer']['distributed']:
        logger.info('Distributed GPU training mode start...')
    else:
        logger.info('One GPU or CPU training mode start...')
    if cfg['trainer']['distributed']:
        dist.init_process_group(backend='nccl', init_method='env://')
        global_rank = dist.get_rank()
        logger.info('Process  {} world size = {}, rank = {}, backend = {}'
                    .format(os.getpid(), dist.get_world_size(),
                            dist.get_rank(), dist.get_backend()))
    main(cfg, logger)
    if cfg['trainer']['distributed']:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='MASTER distributed training')
    parser.add_argument('--config', default='config/master.yaml', help='config path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as stream:
        cfg = yaml.safe_load(stream)
    entry_point(cfg)
