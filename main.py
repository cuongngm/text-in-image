import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.loader.det_loader import DBLoaderTrain
from torch.utils.data import DataLoader
import torch
from src.utils.utils_function import create_module


if __name__ == '__main__':
    with open('config/db_resnet50.yaml', 'r') as stream:
        cfg = yaml.safe_load(stream)
    train_dataset = DBLoaderTrain(cfg)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    samples = next(iter(train_loader))

    db_model = create_module(cfg['model']['function'])(cfg)
    print(db_model)  # <class 'src.model.det_model.DBNet'>
    output = db_model(samples)
    print(output)