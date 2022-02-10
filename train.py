import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import random
import os
import yaml
import argparse
import warnings
import torch

from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from ultocr.logger.logger import setup_logging
from ultocr.loader.base import DistValSampler
from ultocr.loader.recognition.reg_loader import DistCollateFn
from ultocr.utils.utils_function import create_module, str_to_bool
from ultocr.trainer.det_train import TrainerDet
from ultocr.trainer.reg_train import TrainerReg
warnings.filterwarnings('ignore')


def concatenate_dataset(cfg, is_training):
    dataset_list = []
    if is_training:
        root = cfg['dataset']['train_load']['train_root']
        select_data = cfg['dataset']['train_load']['train_select_data']
        if select_data is not None:
            select_data = select_data.split('-')
            for select_d in select_data:
                dataset = create_module(cfg['dataset']['function'])(os.path.join(root, select_d), cfg, is_training=True)
                dataset_list.append(dataset)
    else:
        dataset_list = []
        root = cfg['dataset']['test_load']['test_root']
        select_data = cfg['dataset']['test_load']['test_select_data']
        if select_data is not None:
            select_data = select_data.split('-')
            for select_d in select_data:
                dataset = create_module(cfg['dataset']['function'])(os.path.join(root, select_d), cfg, is_training=False)
                dataset_list.append(dataset)
    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def get_data_loader(cfg):
    if cfg['dataset']['type'] == 'lmdb':
        train_dataset = concatenate_dataset(cfg, is_training=True)
        test_dataset = concatenate_dataset(cfg, is_training=False)
    else:
        train_dataset = create_module(cfg['dataset']['function'])(None, cfg, is_training=True)
        test_dataset = create_module(cfg['dataset']['function'])(None, cfg, is_training=False)
    train_sampler = DistributedSampler(train_dataset) if cfg['trainer']['distributed'] else None
    test_sampler = DistValSampler(list(range(len(test_dataset))), batch_size=cfg['dataset']['test_load']['batch_size'],
                                  distributed=cfg['trainer']['distributed'])
    assert cfg['base']['model_type'] in ['text_detection', 'text_recognition']
    if cfg['base']['model_type'] == 'text_detection':
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg['dataset']['train_load']['batch_size'],
                                  shuffle=False, num_workers=cfg['dataset']['train_load']['num_workers'])
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, batch_size=1,
                                 num_workers=cfg['dataset']['test_load']['num_workers'])
    else:
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg['dataset']['train_load']['batch_size'],
                                  collate_fn=DistCollateFn(), shuffle=False, num_workers=cfg['dataset']['train_load']['num_workers'])
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, batch_size=1,
                                 collate_fn=DistCollateFn(), num_workers=cfg['dataset']['test_load']['num_workers'])
    return train_loader, test_loader


def get_logger(name):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    return logger


def fix_random_seed_for_reproduce(seed):
    # fix random seeds for reproducibility,
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for current gpu
    torch.cuda.manual_seed_all(seed)  # for all gpu
    torch.backends.cudnn.benchmark = False  # if benchmark=True, speed up training, and deterministic will set be False
    torch.backends.cudnn.deterministic = True  # which can slow down training considerably


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

    cfg['trainer']['distributed'] = args.use_dist
    if cfg['trainer']['distributed']:
        local_world_size = args.local_world_size
        cfg['trainer']['local_world_size'] = local_world_size
        cfg['trainer']['local_rank'] = args.local_rank
    else:
        local_world_size = 1
        cfg['trainer']['local_rank'] = args.local_rank
    
    if cfg['trainer']['distributed']:
        if torch.cuda.is_available():
            if torch.cuda.device_count() < local_world_size:
                raise RuntimeError(f'the number of GPU ({torch.cuda.device_count()}) is less than'
                                   f'the number of process ({local_world_size}) running on each node')
            local_check = (args.local_rank == 0)
            logger.info('Distributed GPU training model start...') if local_check else None
        else:
            raise RuntimeError('CUDA is not available, Distributed training is not supported')
    else:
        local_check = True
        logger.info('One GPU or CPU training mode start...') if local_check else None
        if local_world_size != 1:
            raise RuntimeError('local_world_size must set be to 1, if distributed is set to false')
    
    cfg['trainer']['local_check'] = local_check
    fix_random_seed_for_reproduce(123)
    
    if cfg['trainer']['distributed']:
        dist.init_process_group(backend='nccl', init_method='env://')
        logger.info(f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, ' + f'rank = {dist.get_rank()}, backend={dist.get_backend()}') if local_check else None

    # create train, test loader
    train_loader, test_loader = get_data_loader(cfg)
    logger.info('Loaded successful!. Train datasets: {}, test datasets: {}'.format(len(train_loader) * local_world_size * cfg['dataset']['train_load']['batch_size'], len(test_loader) * local_world_size * cfg['dataset']['test_load']['batch_size'])) if local_check else None
    model = create_module(cfg['model']['function'])(cfg)
    logger.info('Model created, trainable parameters:') if local_check else None
    criterion = create_module(cfg['loss']['function'])()
    optimizer = create_module(cfg['optimizer']['function'])(cfg, model.parameters())
    post_process = create_module(cfg['post_process']['function'])(cfg)
    logger.info('Optimizer created.') if local_check else None
    logger.info('Training start...') if local_check else None
    
    assert cfg['base']['model_type'] in ['text_detection', 'text_recognition'], 'dont support this type of model'
    
    if cfg['base']['model_type'] == 'text_detection':
        trainer = TrainerDet(train_loader, test_loader, model, optimizer, criterion, post_process,
                             logger, save_model_dir, cfg)
        trainer.train()
    elif cfg['base']['model_type'] == 'text_recognition':
        trainer = TrainerReg(train_loader, test_loader, model, optimizer, criterion, post_process,
                             logger, save_model_dir, cfg)
        trainer.train()
    if args.device is not None:
        cfg['base']['gpu_id'] = args.device
    logger.info('Training end...') if local_check else None
    if cfg['trainer']['distributed']:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--device', type=str, default=None, help='choose gpu device')
    parser.add_argument('--config', type=str, default='config/db_resnet50.yaml', help='config path')
    parser.add_argument('--local_rank', type=int, default=0, help='automatically passed')
    parser.add_argument('--use_dist', type=str_to_bool, default=True)
    parser.add_argument('--local_world_size', type=int, default=2)
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    main(opt)
