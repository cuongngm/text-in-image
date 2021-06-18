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
    samples = next(iter(train_dataset))
    img = samples[0]

    gt = samples[1]
    gt_mask = samples[2]
    thresh_map = samples[3]
    thresh_mask = samples[4]
    """
    polys = samples[1]
    ignore = samples[2]
    print(img.shape)
    print(polys)
    print(ignore)
    for poly in polys:
        poly = poly.reshape((-1, 1, 2))
        img = cv2.polylines(img, np.int32([poly]), isClosed=True, color=(255, 0, 0), thickness=2)
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(thresh_mask)
    plt.show()