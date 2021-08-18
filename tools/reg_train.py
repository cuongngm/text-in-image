import os
import argparse
import random
import yaml
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.loader.reg_loader import DistCollateFn
from src.utils.utils_function import create_module
from src.utils.logger import TrainLog


def fix_random_seed_for_reproduce(seed):
    # fix random seeds for reproducibility,
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for current gpu
    torch.cuda.manual_seed_all(seed)  # for all gpu
    torch.backends.cudnn.benchmark = False  # if benchmark=True, speed up training, and deterministic will set be False
    torch.backends.cudnn.deterministic = True  # which can slow down training considerably


def main(args, logger):
    with open(args.config, 'r') as stream:
        cfg = yaml.safe_load(stream)
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
    trainer = Trainer(model, optimizer, train_loader, val_loader, lr_scheduler, max_len_step)
    trainer.train()
    logger.info('Distributed training end...')


def entry_point(args, cfg):
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
    main(args, logger)
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
    entry_point(args, cfg)
