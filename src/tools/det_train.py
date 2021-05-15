import cv2
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from src.utils.utils_function import create_dir, create_module


GLOBAL_WORKER_ID = None
GLOBAL_SEED = 123456


def train_val_program(args):
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    create_dir(config['base']['checkpoint'])
    model = create_module(config['model']['function'])(config)
    criterion = create_module(config['loss']['function'])(config)
    train_dataset = create_module(config['train_load']['function'])(config)
    val_dataset = create_module(config['val_load']['function'])(config)
    optimizer = create_module(config['optimizer']['function'])(config, model)
    optimizer_decay = create_module(config['optimizer_decay']['function'])(config)
    img_process = create_module(config['postprocess']['function'])(config)

    train_loader = DataLoader(train_dataset, batch_size=config['train_load']['batch_size'], shuffle=True,
                              num_workers=config['train_load']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['val_load']['batch_size'], shuffle=True,
                            num_workers=config['val_load']['num_workers'])

    if torch.cuda.is_available():
        if len(config['base']['gpu_id'].split(',')) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        criterion.cuda()

    start_epoch = 0
    recall, precision, hmean = 0, 0, 0
    best_recall, best_precision, best_hmean = 0, 0, 0
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--config', help='config path')
    # parameter training
    parser.add_argument('--num_epoch', type=int, default=1200, help='num of epoch training')
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--start_val', type=int, default=400)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--gpd_id', type=int, default=0)
    parser.add_argument()
    args = parser.parse_args()
    train_val_program(args)
