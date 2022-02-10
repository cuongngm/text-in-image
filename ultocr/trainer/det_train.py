import os
import numpy as np
import shutil
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ultocr.utils.utils_function import dict_to_device
from ultocr.metrics.det_metrics import runningScore, cal_text_score, QuadMetric
import mlflow


class TrainerDet:
    def __init__(self, train_loader, test_loader, model, optimizer, criterion, post_process,
                 logger, save_model_dir, config):
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
        self.batch_shape = {'shape': [(config['dataset']['new_shape'][0], config['dataset']['new_shape'][1])]}
        self.metric_cls = QuadMetric()
        self.running_metric_text = runningScore(2)

        self.start_epoch = 1
        self.epochs = config['trainer']['num_epoch']
        
        if config['trainer']['sync_batch_norm'] and self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        if self.distributed:
            self.model = DDP(self.model, device_ids=self.device_ids, output_device=self.device_ids[0],
                             find_unused_parameters=True)
            
        if config['trainer']['resume']:
            assert os.path.isfile(config['trainer']['ckpt_file']), 'checkpoint path is not correct'
            self._resume_checkpoint(config['trainer']['ckpt_file'])

        else:
            logger.info('Training from scratch...') if self.local_check else None

    def train(self):
        self.logger.info('MLflow running...') if self.local_check else None
        if self.distributed:
            dist.barrier()
        best_train_loss = np.inf
        best_hmean = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            torch.cuda.empty_cache()
            self.logger.info('Training in epoch: {}/{}'.format(epoch, self.epochs)) if self.local_check else None
            train_loss = self.train_epoch(epoch)
            train_loss = train_loss.item()
            self.logger.info('Train loss: {}'.format(train_loss)) if self.local_check else None
            # mlflow.log_metric('Train loss', train_loss)
            recall, precision, hmean = self.test_epoch()
            # mlflow.log_metric('Recall', recall)
            # mlflow.log_metric('Precision', precision)
            # mlflow.log_metric('Hmean', hmean)
            
            self.logger.info('Test: Recall: {} - Precision:{} - Hmean: {}'.format(recall, precision, hmean)) if self.local_check else None
            if hmean > best_hmean:
                best_hmean = hmean
                self._save_checkpoint(epoch, save_best=True)
            else:
                self._save_checkpoint(epoch, save_best=False)
        self.logger.info('Saved model') if self.local_check else None

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for idx, batch in enumerate(self.train_loader):
            lr = self.optimizer.param_groups[0]['lr']
            # running_metric_text = self.running_metric_text.reset()
            batch = dict_to_device(batch, device=self.device)
            preds = self.model(batch['img'])
            assert preds.size(1) == 3
            _batch = torch.stack([batch['gt'], batch['gt_mask'],
                                  batch['thresh_map'], batch['thresh_mask']])
            
            total_loss = self.criterion(preds, _batch)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            score_shrink_map = cal_text_score(preds[:, 0, :, :],
                                              batch['gt'], batch['gt_mask'],
                                              self.running_metric_text)
            train_loss += total_loss
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']
            # mlflow.log_param("Batch size", batch['img'].size(0))
            # mlflow.log_param("Learning rate", lr)
            if idx % self.config['trainer']['log_iter'] == 0:
                self.logger.info('[{}-{}] - lr:{} - total-loss:{} - acc:{} - iou:{}'
                                 .format(epoch, idx, lr, total_loss, acc, iou_shrink_map)) if self.local_check else None
        return train_loss / len(self.train_loader)

    def test_epoch(self):
        self.model.eval()
        raw_metrics = []
        for idx, test_batch in tqdm(enumerate(self.test_loader)):
            with torch.no_grad():
                test_batch = dict_to_device(test_batch, device=self.device)
                test_preds = self.model(test_batch['img'])
                box_list, score_list = self.post_process(self.batch_shape, test_preds)
                raw_metric = self.metric_cls.validate_measure(test_batch, (box_list, score_list))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        recall = metrics['recall'].avg
        precision = metrics['precision'].avg
        hmean = metrics['fmeasure'].avg
        return recall, precision, hmean
    
    def prepare_device(self, local_rank, local_world_size):
        if self.distributed:
            ngpu_per_process = torch.cuda.device_count() // local_world_size
            device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))
            
            if torch.cuda.is_available() and local_rank != -1:
                torch.cuda.set_device(device_ids[0])  # device_ids[0] =local_rank if local_world_size = n_gpu per node
                device = 'cuda'
                self.logger.info(
                    f"[Process {os.getpid()}] world_size = {dist.get_world_size()}, "
                    + f"rank = {dist.get_rank()}, n_gpu/process = {ngpu_per_process}, device_ids = {device_ids}"
                ) if self.local_check else None
            else:
                self.logger.warning('Training will be using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, device_ids
        else:
            n_gpu = torch.cuda.device_count()
            n_gpu_use = 1
            if n_gpu_use > 0 and n_gpu == 0:
                self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                    "training will be performed on CPU.")
                n_gpu_use = 0
            if n_gpu_use > n_gpu:
                self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                    "on this machine.".format(n_gpu_use, n_gpu))
                n_gpu_use = n_gpu

            list_ids = list(range(n_gpu))
            if n_gpu_use > 0:
                torch.cuda.set_device(self.config['base']['gpu_id'])  # only use first available gpu as devices
                self.logger.warning(f'Training is using GPU {self.config["base"]["gpu_id"]}!')
                device = 'cuda'
            else:
                self.logger.warning('Training is using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, step_idx=None):
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
        if step_idx is None:
            filename = str(self.save_model_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        else:
            filename = str(self.save_model_dir / 'checkpoint-epoch{}-step{}.pth'.format(epoch, step_idx))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename)) if self.local_check else None

        if save_best:
            best_path = str(self.save_model_dir / 'model_best.pth')
            shutil.copyfile(filename, best_path)
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