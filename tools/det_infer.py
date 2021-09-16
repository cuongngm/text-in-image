import os
import cv2
import argparse
import yaml
import numpy as np
import torch
import sys
import time
from PIL import Image
import torchvision.transforms as transforms
sys.path.append('./')
from ultocr.utils.utils_function import create_module, load_model, resize_image_batch, create_dir


def config_load(args):
    stream = open(args.config, 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    config['infer']['model_path'] = args.model_path
    config['infer']['img_path'] = args.img_path
    config['infer']['result_path'] = args.result_path
    return config


def get_batch_files(path, img_files, batch_size=3):
    img_files = np.array(img_files)
    num = len(img_files)//batch_size
    batch_imgs = []
    batch_img_names = []
    for i in range(num):
        files = img_files[batch_size*i:batch_size*(i+1)]
        img = [cv2.imread(os.path.join(path, img_file)) for img_file in files]
        img_names = [img_file.split('.')[0] for img_file in files]
        batch_imgs.append(img)
        batch_img_names.append(img_names)
    files = img_files[batch_size*(num):len(img_files)]
    if len(files) != 0:
        img = [cv2.imread(os.path.join(path, img_file)) for img_file in files]
        img_names = [img_file.split('.')[0] for img_file in files]
        batch_imgs.append(img)
        batch_img_names.append(img_names)
    return batch_imgs, batch_img_names


def get_img(ori_imgs, config):
    imgs = []
    scales = []
    for ori_img in ori_imgs:
        img, scale = resize_image_batch(ori_img, config['val_load']['test_size'], add_padding=False)
        img = Image.fromarray(img).convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0)
        imgs.append(img)
        scales.append(scale)
    return torch.cat(imgs, 0), scales


class DetectDB:
    def __init__(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        model = create_module(config['model']['function'])(config)
        model = load_model(model, config['infer']['model_path'], self.device)
        self.model = model.to(self.device)
        self.thresh = config['postprocess']['thresh']
        self.box_thresh = config['postprocess']['box_thresh']
        self.unclip_ratio = config['postprocess']['unclip_ratio']
        self.img_process = create_module(config['postprocess']['function'])(config)

    def detect(self, img_path):
        img_ori = cv2.imread(img_path)
        h_origin, w_origin = img_ori.shape[:2]
        img, scales = get_img(img_ori, self.config)
        img = img.to(self.device)
        with torch.no_grad():
            out = self.model(img)
        out = out.cpu().numpy()
        bbox_batch, score_batch = self.img_process(out, scales)
        bboxes = bbox_batch[0]
        for bbox in bboxes:
            bbox = bbox.reshape(-1, 2)

        return img


class TestProgram:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = create_module(config['model']['function'])(config)
        model = load_model(model, config['infer']['model_path'])
        model.to(self.device)
        self.model = model
        self.model.eval()
        img_process = create_module(config['postprocess']['function'])(config)
        self.img_process = img_process

    def infer_img(self, ori_imgs):
        img, scales = get_img(ori_imgs, self.config)
        img = img.to(self.device)
        with torch.no_grad():
            out = self.model(img)
        out = out.cpu().numpy()
        bbox_batch, score_batch = self.img_process(out, scales)
        return bbox_batch, score_batch


def infer_one_img(bin, img, img_name, result_path):
    bbox_batch, score_batch = bin.infer_img(img)
    for i in range(len(bbox_batch)):
        img_show = img[i].copy()
        with open(os.path.join(result_path, 'result_txt', 'res_' + img_name[i] + '.txt'), 'w', encoding='utf-8') as fid_res:
            bboxes = bbox_batch[i]
            for bbox in bboxes:
                bbox = bbox.reshape(-1, 2).astype(np.int)
                img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
                bbox_str = [str(x) for x in bbox.reshape(-1)]
                bbox_str = ','.join(bbox_str) + '\n'
                fid_res.write(bbox_str)
        cv2.imwrite(os.path.join(result_path, 'result_img', img_name[i] + '.jpg'), img_show)


def infer_image(config):
    img_path = config['infer']['img_path']
    result_path = config['infer']['result_path']
    test_bin = TestProgram(config)
    create_dir(result_path)
    create_dir(os.path.join(result_path, 'result_img'))
    create_dir(os.path.join(result_path, 'result_txt'))
    if os.path.isdir(img_path):
        print('inference with image folder')
        files = os.listdir(img_path)
        batch_imgs, batch_img_names = get_batch_files(img_path, files, batch_size=config['val_load']['batch_size'])
        for i in range(len(batch_imgs)):
            infer_one_img(test_bin, batch_imgs[i], batch_img_names[i], result_path)
    else:
        print('inference with image path')
        image_name = img_path.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path)
        infer_one_img(test_bin, [img], [image_name], result_path)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper params')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--result_path', type=str, default=None)
    args = parser.parse_args()
    config = config_load(args)
    exe_time = time.time()
    infer_image(config)
    print('Execution time:', time.time() - exe_time)
