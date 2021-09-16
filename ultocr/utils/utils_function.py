import os
import cv2
import numpy as np
import random
import json
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import importlib
import torch


def cv2pillow(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def pillow2cv2(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def create_module(module_str):
    tmpss = module_str.split(',')
    assert len(tmpss) == 2, 'Error format of the module path: {}'.format(module_str)
    module_name, function_name = tmpss[0], tmpss[1]
    somemodule = importlib.import_module(module_name, __package__)
    function = getattr(somemodule, function_name)
    return function


def save_checkpoint(state, checkpoint='checkpoint', filename='model_best.pth'):
    model_path = os.path.join(checkpoint, filename)
    torch.save(state, model_path)


def load_model(model, model_path, device='cuda:0'):
    model_dict = torch.load(model_path, map_location=device)
    if 'state_dict' in model_dict.keys():
        model_dict = model_dict['state_dict']
    try:
        model.load_state_dict(model_dict)
    except:
        state = model.state_dict()
        for key in state.keys():
            state[key] = model_dict['module.' + key]
        model.load_state_dict(state)
    return model


def sed_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dir(path):
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def create_loss_bin():
    bin_dict = {}
    keys = ['loss_total', 'loss_l1', 'loss_bce', 'loss_thresh']
    for key in keys:
        bin_dict[key] = LossAccumulator()
    return bin_dict


class LossAccumulator:
    def __init__(self):
        super().__init__()
        self.loss_items = []

    def loss_add(self, loss):
        self.loss_items.append(loss)

    def loss_sum(self):
        return sum(self.loss_items)

    def loss_mean(self):
        return sum(self.loss_items) / len(self.loss_items)

    def loss_clear(self):
        self.loss_items = []


def merge_config(config, args):
    for key_1 in config.keys():
        if isinstance(config[key_1], dict):
            for key_2 in config[key_1].keys():
                if key_2 in dir(args):
                    config[key_1, key_2] = getattr(args, key_2)
    return config


def dict_to_device(batch, device='cuda'):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def read_json(filename):
    filename = Path(filename)
    with filename.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)
