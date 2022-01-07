import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import os
import yaml
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from ultocr.logger.logger import setup_logging
from ultocr.utils.utils_function import create_module
from ultocr.trainer.det_train import TrainerDet
from ultocr.trainer.reg_train import TrainerReg
warnings.filterwarnings('ignore')


def get_data_loader(cfg, logger):
    train_dataset = create_module(cfg['dataset']['function'])(cfg, is_training=True)
    test_dataset = create_module(cfg['dataset']['function'])(cfg, is_training=False)
    if cfg['trainer']['distributed']:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg['dataset']['train_load']['batch_size'],
                                  shuffle=False, num_workers=cfg['dataset']['train_load']['num_workers'])
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=cfg['dataset']['test_load']['batch_size'],
                                 shuffle=False, num_workers=cfg['dataset']['test_load']['num_workers'])
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg['dataset']['train_load']['batch_size'],
                                  shuffle=False, num_workers=cfg['dataset']['train_load']['num_workers'])
        test_loader = DataLoader(test_dataset,
                                 batch_size=cfg['dataset']['test_load']['batch_size'],
                                 shuffle=False, num_workers=cfg['dataset']['test_load']['num_workers'])
    logger.info('Loaded successful!. Train datasets: {}, test datasets: {}'.format(len(train_dataset), len(test_dataset)))
    return train_loader, test_loader


def get_logger(name):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    return logger


def main(args):
    with open(args.config, 'r') as stream:
        cfg = yaml.safe_load(stream)
    # create log and save file
    save_dir = Path(cfg['base']['save_dir'])
    exp_name = cfg['base']['algorithm']
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    save_model_dir = save_dir / 'ckpt' / exp_name / run_id
    log_dir = save_dir / 'logs' / exp_name / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    save_model_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir)
    logger = get_logger('train')

    # local_rank = args.local_rank
    local_world_size = cfg['trainer']['local_world_size']
    if cfg['trainer']['distributed']:
        logger.info('Distributed GPU training model start...')
        if torch.cuda.is_available():
            if torch.cuda.device_count() < local_world_size:
                raise RuntimeError(f'the number of GPU ({torch.cuda.device_count()}) is less than'
                                   f'the number of process ({local_world_size}) running on each node')
            
        else:
            raise RuntimeError('CUDA is not available, Distributed training is not supported')
    else:
        logger.info('One GPU or CPU training mode start...')
        if local_world_size != 1:
            raise RuntimeError('local_world_size must set be to 1, if distributed is set to false')
        local_rank = 0
        global_rank = 0
    
    if cfg['trainer']['distributed']:
        dist.init_process_group(backend='nccl', init_method='env://')
        global_rank = dist.get_rank()
        logger.info(f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, ' + f'rank = {dist.get_rank()}, backend={dist.get_backend()}')

    # create train, test loader
    train_loader, test_loader = get_data_loader(cfg, logger)

    model = create_module(cfg['model']['function'])(cfg)
    logger.info('Model created, trainable parameters:')
    criterion = create_module(cfg['loss']['function'])(cfg['loss']['l1_scale'], cfg['loss']['bce_scale'])
    optimizer = create_module(cfg['optimizer']['function'])(cfg, model.parameters())
    post_process = create_module(cfg['post_process']['function'])(cfg)
    logger.info('Optimizer created.')
    logger.info('Training start...')
    
    assert cfg['base']['model_type'] in ['text_detection', 'text_recognition'], 'dont support this type of model'
    if cfg['base']['model_type'] == 'text_detection':
        trainer = TrainerDet(train_loader, test_loader, model, optimizer, criterion, post_process,
                             logger, save_model_dir, cfg)
        trainer.train()
    elif cfg['base']['model_type'] == 'text_recognition':
        trainer = TrainerReg(train_loader, test_loader, model, optimizer, criterion, post_process,
                             logger, save_model_dir, cfg)
        trainer.train()

    logger.info('Training end...')
    if cfg['trainer']['distributed']:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--config', type=str, default='config/db_resnet50.yaml', help='config path')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    main(opt)

