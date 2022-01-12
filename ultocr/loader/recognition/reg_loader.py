import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from ultocr.loader.recognition.translate import LabelTransformer
from ultocr.utils.utils_function import create_module


class RegLoader(Dataset):
    def __init__(self, config, is_training=True):
        self.img_w = config['dataset']['new_shape'][1]
        self.img_h = config['dataset']['new_shape'][0]
        
        self.case_sensitive = config['dataset']['preprocess']['case_sensitive']
        self.to_gray = config['dataset']['preprocess']['to_gray']
        self.transform = create_module(config['dataset']['preprocess']['transform'])(self.img_h, self.img_w)

        if is_training:
            images, labels = self.get_base_info(config['dataset']['train_load']['train_img_dir'],
                                                config['dataset']['train_load']['train_label_dir'])
        else:
            images, labels = self.get_base_info(config['dataset']['test_load']['test_img_dir'],
                                                config['dataset']['test_load']['test_label_dir'])
        self.all_images = images
        self.all_labels = labels
        self.is_training = is_training

    def get_base_info(self, img_root, txt_file):
        image_names = []
        labels = []
        with open(txt_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                image_name = line[0]
                label = '\n'.join(line[1:])
                if len(label) > LabelTransformer.max_length and LabelTransformer.max_length != -1:
                    continue
                image_name = os.path.join(img_root, image_name)
                image_names.append(image_name)
                labels.append(label)
        return image_names, labels

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        file_name = self.all_images[idx]
        img = Image.open(file_name)
        try:
            if self.to_gray:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
        except Exception as e:
            print('Error image for {}'.format(file_name))

        if self.transform is not None:
            img, width_ratio = self.transform(img)
            
        if self.is_training:
            label = self.all_labels[idx]

            if not self.case_sensitive:
                label = label.lower()
            return img, label
        else:
            return img, file_name


class TextInference(Dataset):
    def __init__(self, all_img, transform=None):
        self.all_img = all_img
        self.transform = transform

    def __getitem__(self, idx):
        img = self.all_img[idx]
        if self.transform is not None:
            img, width_ratio = self.transform(img)
            return img

    def __len__(self):
        return len(self.all_img)


class DistCollateFn:
    def __init__(self, training=True):
        self.training = training

    def __call__(self, batch):
        batch_size = len(batch)
        if batch_size == 0:
            return dict(batch_size=batch_size, images=None, labels=None)

        if self.training:
            images, labels = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return dict(batch_size=batch_size,
                        images=image_batch_tensor,
                        labels=labels)
        else:
            images, file_names = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return dict(batch_size=batch_size,
                        images=image_batch_tensor,
                        file_names=file_names)


class Resize:
    def __init__(self, new_h, new_w, is_gray=False):
        self.new_h = new_h
        self.new_w = new_w
        self.is_gray = is_gray

    def __call__(self, img:Image.Image):
        if self.is_gray:
            img = img.convert('L')
        img = np.array(img)
        h, w = img.shape[:2]
        resize_img = cv2.resize(img, (self.new_w, self.new_h))
        full_channel_img = resize_img[..., None] if len(resize_img.shape) == 2 else resize_img
        resize_img_tensor = torch.from_numpy(np.transpose(full_channel_img, (2, 0, 1))).to(torch.float32)
        resize_img_tensor.sub_(127.5).div_(127.5)
        return resize_img_tensor, w/self.new_w
