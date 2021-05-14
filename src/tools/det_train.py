import cv2
import yaml
import argparse
import torch


GLOBAL_WORKER_ID = None
GLOBAL_SEED = 123456


def train_val_program(args):
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

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
