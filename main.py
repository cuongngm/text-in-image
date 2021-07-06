import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.loader.det_loader import DBLoader
from torch.utils.data import DataLoader
import torch
from src.utils.utils_function import create_module
from src.model.det_model.db_net import DBNetVer1


if __name__ == '__main__':
    with open('config/db_resnet50.yaml', 'r') as stream:
        cfg = yaml.safe_load(stream)

    train_dataset = DBLoader(cfg, img_dir=cfg['train_load']['train_img_dir'],
                             label_dir=cfg['train_load']['train_label_dir'], is_training=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    # samples = next(iter(train_dataset))
    samples = train_dataset[28]
    img = samples[0]

    gt = samples[1]
    gt_mask = samples[2]

    thresh_map = samples[3]
    thresh_mask = samples[4]
    cv2.imshow('img', img)
    cv2.imshow('gt', gt)
    cv2.imshow('thresh map', thresh_map)
    cv2.waitKey(0)
    """
    polys = samples[1]
    ignore = samples[2]
    print(img.shape)
    print(polys)
    print(ignore)
    for poly in polys:
        poly = poly.reshape((-1, 1, 2))
        img = cv2.polylines(img, np.int32([poly]), isClosed=True, color=(255, 0, 0), thickness=2)
    
    fig = plt.figure(figsize=(10, 10))
    rows = 2
    columns = 2
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    # showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("First")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    # showing image
    plt.imshow(gt)
    plt.axis('off')
    plt.title("Second")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    # showing image
    plt.imshow(thresh_map)
    plt.axis('off')
    plt.title("Third")
    
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
    # showing image
    plt.imshow(thresh_mask)
    plt.axis('off')
    plt.title("Fourth")
    # plt.imshow(img)

    plt.show()
    """
