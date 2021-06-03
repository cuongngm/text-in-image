import os
import cv2
import argparse
import yaml
import numpy as np
import torch
import sys
sys.path.append('./')
from src.utils.utils_function import create_module, load_model


def config_load(args):
    stream = open(args.config, 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    config['infer']['model_path'] = args.model_path
    config['infer']['img_path'] = args.img_path
    config['infer']['result_path'] = args.result_path
    return config


def infer_image(config):
    img_path = config['infer']['img_path']
    result_path = config['infer']['result_path']
    return config


class TestProgram:
    def __init__(self, config):
        super().__init__()
        self.config = config
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = create_module(config['model']['function'])(config)
        model = load_model(model, config['infer']['model_path'])
        model.to(device)
        self.model = model
        self.model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--result_path', type=str, default=None)
    args = parser.parse_args()
    config = config_load(args)
    infer_image(config)
