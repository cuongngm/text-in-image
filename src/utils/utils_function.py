import os
import cv2
import numpy as np
import random
from PIL import Image
from pathlib import Path
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


def sed_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dir(path):
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)