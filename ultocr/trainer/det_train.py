import numpy as np
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from ultocr.utils.utils_function import create_module, save_checkpoint, dict_to_device
from ultocr.utils.det_metrics import runningScore, cal_text_score, QuadMetric


class TrainerDet:
    def __init__(self, train_loader, test_loader, model, optimizer, criterion, post_process,
                 logger, save_model_dir, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.distributed = config['trainer']['distributed']
        if self.distributed:
            self.local_master = (config['trainer']['local_rank'] == 0)
            self.global_master = (dist.get_rank() == 0)
        else:
            self.local_master = True
            self.global_master = True

        
        self.logger = logger
        self.save_model_dir = save_model_dir
        self.device, self.device_ids = self.prepare_device(config['trainer']['local_rank'], config['trainer']['local_world_size'])
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.post_process = post_process
        self.batch_shape = {'shape': [(config['dataset']['crop_shape'][0], config['dataset']['crop_shape'][1])]}
        self.metric_cls = QuadMetric()
        self.running_metric_text = runningScore(config['trainer']['num_class'])

        self.start_epoch = 1
        self.epochs = config['trainer']['num_epoch']
        self.config = config
        # if config['trainer']['sync_batch_norm'] and self.distributed:
        #     self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        if self.distributed:
            self.model = DDP(self.model, device_ids=self.device_ids, output_device=self.device_ids[0],
                             find_unused_parameters=True)

        if config['base']['resume']:
            assert os.path.isfile(config['base']['ckpt_file']), 'checkpoint path is not correct'
            logger.info('Resume from checkpoint: {}'.format(config['base']['ckpt_file']))
            checkpoint = torch.load(config['base']['ckpt_file'])
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            logger.info('Training from scratch...')

    def train(self):
        if self.distributed:
            dist.barrier()
        best_train_loss = np.inf
        best_hmean = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            torch.cuda.empty_cache()
            self.logger.info('Training in epoch: {}/{}'.format(epoch, self.epochs))
            train_loss = self.train_epoch(epoch)
            train_loss = train_loss.item()
            self.logger.info('Train loss: {}'.format(train_loss))
            recall, precision, hmean = self.test_epoch()
            self.logger.info('Test: Recall: {} - Precision:{} - Hmean: {}'.format(recall, precision, hmean))
            if hmean > best_hmean:
                best_hmean = hmean
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict()
                }, self.save_model_dir, 'best_hmean.pth')
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict()
                }, self.save_model_dir, 'best_cp.pth')
        self.logger.info('Training completed')
        save_checkpoint({
            'epoch': self.epochs,
            'state_dict': self.model.state_dict()
        }, self.save_model_dir, 'last_cp.pth')
        self.logger.info('Saved model')
        if self.distributed:
            dist.destroy_process_group()

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
            if idx % self.config['trainer']['log_iter'] == 0:
                self.logger.info('[{}-{}] - lr:{} - total-loss:{} - acc:{} - iou:{}'
                                 .format(epoch, idx, lr, total_loss, acc, iou_shrink_map))
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
            print('device_ids', device_ids)
            if torch.cuda.is_available() and local_rank != -1:
                torch.cuda.set_device(device_ids[0])
                device = 'cuda'
                self.logger.info(f"[Process {os.getpid()}] world_size = {dist.get_world_size()}, " +
                                 f"rank = {dist.get_rank()}, n_gpu/process = {ngpu_per_process}," +
                                 f"device_ids = {device_ids}")
            else:
                self.logger.warning('Training will be using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, device_ids
        else:
            n_gpu = torch.cuda.device_count()
            n_gpu_use = local_world_size
            if n_gpu_use > 0 and n_gpu == 0:
                self.logger.warning('Warning: There\'s no GPU available on this machine,'
                                    'training will be performed on CPU.')
                n_gpu_use = 0
            if n_gpu_use > n_gpu:
                self.logger.warning('Warning: The number of GPU\'s configured to use is {},'
                                    'but only {} is available'.format(n_gpu_use, n_gpu))
                n_gpu_use = n_gpu
            list_ids = list(range(n_gpu))
            if n_gpu_use > 0:
                torch.cuda.set_device(list_ids[1])
                self.logger.warning(f'Training is using GPU {list_ids[1]}!')
                device = 'cuda'
            else:
                self.logger.warning('Training is using GPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, list_ids
