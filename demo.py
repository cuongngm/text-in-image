import yaml
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader
from ultocr.loader.detection.det_loader import DetLoader
from ultocr.utils.utils_function import read_json
from ultocr.inference import End2end


def visualize_dbnet():
    with open('config/db_resnet50.yaml', 'r') as stream:
        cfg = yaml.safe_load(stream)

    train_dataset = DetLoader(cfg, is_training=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    # test_dataset = DetLoader(cfg, is_training=False)
    # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    # sample = next(iter(train_loader))

    samples = train_dataset[22]
    img = samples['img']
    gt = samples['gt']
    gt_mask = samples['gt_mask']
    thresh_map = samples['thresh_map']
    thresh_mask = samples['thresh_mask']
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


def parse_args():
    parser = argparse.ArgumentParser(description='Hyper parameter')
    parser.add_argument('--image_path', type=str, default='saved/test/input/1.jpg', help='input image path')
    parser.add_argument('--save_path', type=str, default='saved/test/output/', help='save path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import time
    opt = parse_args()
    img_path = opt.image_path
    img_name = img_path.split('/')[-1]
    result_name = img_name[:-4] + '_rs' + '.jpg'
    model = End2end(img_path)
    predict_time = time.time()
    result, img_rs = model.get_result()
    print('predict time:', time.time() - predict_time)
    # img = result['img']
    cv2.imwrite(os.path.join(opt.save_path, result_name), img_rs)
    print(result)
