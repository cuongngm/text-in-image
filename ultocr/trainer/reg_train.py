from collections import OrderedDict
import os
import shutil
from tqdm import tqdm
import distance
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ultocr.loader.recognition.translate import LabelConverter
from ultocr.metrics.reg_metrics import AverageMetricTracker
from ultocr.utils.utils_function import create_module
# from ultocr.model.recognition.postprocess import greedy_decode
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.models.signature import infer_signature


class TrainerReg:
    def __init__(self, train_loader, test_loader, model, optimizer, criterion, post_process, logger, save_model_dir, config):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.distributed = config['trainer']['distributed']
        if self.distributed:
            self.local_check = (config['trainer']['local_check'] == 0)
        else:
            self.local_check = True
        self.logger = logger
        self.save_model_dir = save_model_dir
        self.device, self.device_ids = self.prepare_device(config['trainer']['local_rank'],
                                                           config['trainer']['local_world_size'])
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.post_process = post_process
        self.convert = LabelConverter(classes=config['dataset']['vocab'], max_length=config['post_process']['max_len'], ignore_over=False)
        self.train_metrics = AverageMetricTracker('loss')
        self.val_metrics = AverageMetricTracker('loss', 'word_acc', 'word_acc_case_insensitive',
                                                'edit_distance_acc')
        self.start_epoch = 1
        self.epochs = config['trainer']['epochs']
        self.resume = config['trainer']['resume']

        self.monitor_best = 0
        
        if self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=self.device_ids, output_device=self.device_ids[0],
                             find_unused_parameters=True)
        # resume from checkpoint
        if config['trainer']['resume']:
            assert os.path.isfile(config['trainer']['ckpt_file']), 'checkpoint path is not correct'
            self._resume_checkpoint(config['trainer']['ckpt_file'])

        else:
            logger.info('Training from scratch...') if self.local_check else None
            
        self.len_step = len(train_loader)
        self.do_validation = (self.test_loader is not None and config['trainer']['do_validation'])
        self.validation_epoch = config['trainer']['validation_epoch']
        self.lr_scheduler = None
        log_step = config['trainer']['log_step_interval']
        self.log_step = log_step
        self.val_step_interval = config['trainer']['val_step_interval']
        self.early_stop = config['trainer']['early_stop']  

    def train(self):
        # client = MlflowClient()
        # artifact_path = 'master-model'
        # mlflow.set_tracking_uri("sqlite:///mlruns.db")
     
        # with mlflow.start_run() as run:
        #     run_num = run.info.run_id
        #     mlflow.log_param("Experiment name", run_num)
        
        if self.distributed:
            dist.barrier()  # syncing machines before training
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            # self.test_loader.batch_sampler.set_epoch(epoch) if self.test_loader.batch_sampler is not None else None
            
            torch.cuda.empty_cache()
            result_dict = self.train_epoch(epoch)
            if self.do_validation and epoch % self.validation_epoch == 0:
                val_metric_res_dict = self.valid_epoch(epoch)
                val_res = f"\nValidation result after {epoch} epoch:" \
                          f"Word_acc: {val_metric_res_dict['word_acc']:.6f}" \
                          f"Word_acc_case_ins: {val_metric_res_dict['word_acc_case_insensitive']:.6f}" \
                          f"Edit_distance_acc: {val_metric_res_dict['edit_distance_acc']:.6f}"
                
                # mlflow.log_metric('word_acc', val_metric_res_dict['word_acc'])
                # mlflow.log_metric('word_acc_case_ins', val_metric_res_dict['word_acc_case_insensitive'])
                # mlflow.log_metric('edit_distance_acc', val_metric_res_dict['edit_distance_acc'])
            else:
                val_res = ''
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.logger.info('[Epoch end] Epoch:[{}/{}] Loss: {:.6f} LR: {:.8f}'
                             .format(epoch, self.epochs, result_dict['loss'], self.get_lr()) + val_res) if self.local_check else None
            # mlflow.log_metric('loss', result_dict['loss'])
           
            best = False
            if self.do_validation and epoch % self.validation_epoch == 0:
                
                best, not_improved_count = self.is_best_monitor_metric(best, not_improved_count, val_metric_res_dict)
                if not_improved_count > self.early_stop:
                    self.logger.info('Validation performance didn\'t improve for {} epochs.'
                                     'Training stops'.format(self.early_stop)) if self.local_check else None
                    break
                if best:
                    self._save_checkpoint(epoch, save_best=True, save_current=False)
                else:
                    self._save_checkpoint(epoch, save_best=False, save_current=False)
        self.logger.info('Saved model') if self.local_check else None
       
        # model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_num, artifact_path=artifact_path)
        # print(model_uri)
        # mlflow.pytorch.log_model(self.model, artifact_path)
        # mlflow.pytorch.save_model(self.model, artifact_path)
        # mlflow.pytorch.save_state_dict(self.model.state_dict(), artifact_path)
        # mlflow.register_model(model_uri=model_uri, name=artifact_path)
    
    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for step_idx, input_data_item in enumerate(self.train_loader):
            images = input_data_item[0]
            if images is None: continue
            text_label = input_data_item[1]
            step_idx += 1
            images = images.to(self.device)
            target = self.convert.encode(text_label)
            target = target.to(self.device)
            target = target.permute(1, 0)
            with torch.autograd.set_detect_anomaly(True):
                outputs = self.model(images, target[:, :-1])
                loss = self.criterion(outputs.contiguous().view(-1, outputs.shape[-1]),
                                       target[:, 1:].contiguous().view(-1), ignore_index=self.convert.PAD)
               
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_total = images.shape[0]
            reduced_loss = loss.item()
            if self.distributed:
                reduced_metrics_tensor = torch.tensor([batch_total, reduced_loss]).float().to(self.device)
                dist.barrier()
                reduced_metrics_tensor = self.sum_tensor(reduced_metrics_tensor)
                batch_total, reduced_loss = reduced_metrics_tensor.cpu().numpy()
                reduced_loss = reduced_loss / dist.get_world_size()
            global_step = (epoch - 1) * self.len_step + step_idx - 1
            self.train_metrics.update('loss', reduced_loss, batch_total)
            if step_idx % self.log_step == 0 or step_idx == 1:
                self.logger.info('Train Epoch:[{}/{}] Step:[{}/{}] Loss:{:.6f} Loss avg:{:.6f} LR:{:.8f}'
                                 .format(epoch, self.epochs, step_idx, self.len_step,
                                         self.train_metrics.val('loss'),
                                         self.train_metrics.avg('loss'),
                                         self.get_lr())) if self.local_check else None
            if step_idx == self.len_step:
               break
        log_dict = self.train_metrics.result()
        return log_dict

    def valid_epoch(self, epoch):
        self.model.eval()
        self.val_metrics.reset()
        for step_idx, input_data_item in tqdm(enumerate(self.test_loader)):
            images = input_data_item[0]
            if images is None: continue
            batch_size = images.size(0)
            text_label = input_data_item[1]
            encode_label = self.convert.encode(text_label)
            
            # if self.distributed:
            #     word_acc, word_acc_case_ins, edit_distance_acc, total_distance_ref, batch_total =\
            #         self.distributed_predict(batch_size, images, text_label)
            # else:
            with torch.no_grad():
                images = images.to(self.device)

                if hasattr(self.model, 'module'):
                    model = self.model.module
                else:
                    model = self.model

            # outputs = greedy_decode(model, images, 100, 2, 0, self.device, True)
            outputs = self.post_process.greedy_decode(model, images, self.config['post_process']['max_len'], 2, 0, self.device, True)
            correct = 0
            correct_case_ins = 0
            total_distance_ref = 0
            total_edit_distance = 0
            for index, (pred, text_gold) in enumerate(zip(outputs[:, 1:], text_label)):
                predict_text = ''
                for i in range(len(pred)):
                    if pred[i] == self.convert.EOS: break
                    if pred[i] == self.convert.UNK: continue
                    decoded_char = self.convert.decode(pred[i])
                    predict_text += decoded_char
                # print('gt:', text_gold)
                # print('pred:', predict_text)
                
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
                outputs = greedy_decode(model, images, 100, 2, 0, self.device, True)
                for idx, (pred, text_gold) in enumerate(zip(outputs[:, 1:], text_label)):
                    predict_text = ''
                    for i in range(len(pred)):
                        if pred[i] == self.convert.EOS: break
                        if pred[i] == self.convert.UNK: continue
                        decoded_char = self.convert.decode(pred[i])
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

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group['lr']

    def prepare_device(self, local_rank, local_world_size):
        if self.distributed:
            ngpu_per_process = torch.cuda.device_count() // local_world_size
            device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))
            if torch.cuda.is_available() and local_rank != -1:
                torch.cuda.set_device(device_ids[0])
                device = 'cuda'
                self.logger.info('[Process {}] world size = {}, rank = {}, n_gpu/process = {}, device_ids = {}'
                                 .format(os.getpid(), dist.get_world_size(),
                                         dist.get_rank(), ngpu_per_process, device_ids)) if self.local_check else None
            else:
                self.logger.info('Training will be using CPU!') if self.local_check else None
                device = 'cpu'
            device = torch.device(device)
            return device, device_ids
        else:
            n_gpu = torch.cuda.device_count()
            n_gpu_use = 1
            if n_gpu_use > 0 and n_gpu == 0:
                self.logger.info('Warning: There is no GPU available on this machine,'
                                 'training will be performed on CPU') if self.local_check else None
                n_gpu_use = 0
            if n_gpu_use > n_gpu:
                self.logger.info('Warning: The number of GPU configured to use is {}, but only {} are'
                                 'available on this machine'.format(n_gpu_use, n_gpu)) if self.local_check else None
                n_gpu_use = n_gpu
            list_ids = list(range(n_gpu_use))
            if n_gpu_use > 0:
                torch.cuda.set_device(self.config['base']['gpu_id'])
                self.logger.info('Training is using GPU {}'.format(self.config["base"]["gpu_id"])) if self.local_check else None
                device = 'cuda'
            else:
                self.logger.warning('Training is using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, list_ids
        
    def _save_checkpoint(self, epoch, save_best=False, save_current=False):
        '''
        Saving checkpoints
        :param epoch:  current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        :return:
        '''
        if hasattr(self.model, 'module'):
            arch_name = type(self.model.module).__name__
            model_state_dict = self.model.module.state_dict()
        else:
            arch_name = type(self.model).__name__
            model_state_dict = self.model.state_dict()
        state = {
            'arch': arch_name,
            'epoch': epoch,
            'model_state_dict': model_state_dict,
        }
        filename = str(self.save_model_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        if save_current:
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename)) if self.local_check else None

        if save_best:
            best_path = str(self.save_model_dir / 'model_best.pth')
            # shutil.copyfile(filename, best_path)
            torch.save(state, best_path)
            self.logger.info(
                f"Saving current best (at {epoch} epoch): model_best.pth") if self.local_check else None

    def _resume_checkpoint(self, resume_path):
        '''
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        :return:
        '''
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path)) if self.local_check else None
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config['local_rank']}
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        # self.monitor_best = checkpoint['monitor_best']

        state_dict = checkpoint['model_state_dict']
        if self.distributed:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)) if self.local_check else None
