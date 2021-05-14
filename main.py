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
    # train_dataset = DBLoaderTrain(cfg)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    db_model = create_module(cfg['model']['function'])
    img = torch.randn(1, 3, 640, 640)
    output = db_model(img)
    print(output.size())
    """
    img, gt, gt_mask, thresh, thresh_mask = next(iter(train_loader))
    print(img.size())
    print(gt.size())
    print(gt_mask.size())
    print(thresh.size())
    print(thresh_mask.size())
    
    for idx, label in enumerate(labels):
        pts = np.array(label, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    
    cv2.imshow('img', gt[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """