import os
import lmdb
import io
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from ultocr.utils.utils_function import create_module


class RegLoader(Dataset):
    def __init__(self, root, config, is_training=True):
        self.img_w = config['dataset']['new_shape'][0]
        self.img_h = config['dataset']['new_shape'][1]
        self.max_length = config['post_process']['max_len']
        self.case_sensitive = config['dataset']['preprocess']['case_sensitive']
        self.to_gray = config['dataset']['preprocess']['to_gray']
        self.transform = create_module(config['dataset']['preprocess']['transform'])(self.img_w, self.img_h)
        if config['dataset']['type'] == 'txt':
            if is_training:
                images, labels = self.get_base_info(config['dataset']['train_load']['train_img_dir'],
                                                    config['dataset']['train_load']['train_label_dir'])
            else:
                images, labels = self.get_base_info(config['dataset']['test_load']['test_img_dir'],
                                                    config['dataset']['test_load']['test_label_dir'])
            self.all_images = images
            self.all_labels = labels
        
        if config['dataset']['type'] == 'lmdb':
            self.env = lmdb.open(root,
                    max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            if not self.env:
                raise RuntimeError('Lmdb file cannot be open')
            self.all_images, self.all_labels = self.get_base_info_lmdb()
        self.is_training = is_training
        self.config = config
        
    def get_base_info_lmdb(self):
        image_keys = []
        labels = []
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b"num-samples").decode())
            for i in range(nSamples):
                index = i + 1
                image_key = ('image-%09d' % index).encode()
                label_key = ('label-%09d' % index).encode()

                label = txn.get(label_key).decode()

                if len(label) > self.max_length and self.max_length != -1:
                    continue

                image_keys.append(image_key)
                labels.append(label)
        return image_keys, labels
    
    def get_base_info(self, img_root, txt_file):
        image_names = []
        labels = []
        with open(txt_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                image_name = line[0]
                label = '\n'.join(line[1:])
                if (len(label) > self.max_length) and (self.max_length != -1):
                    continue
                image_name = os.path.join(img_root, image_name)
                image_names.append(image_name)
                labels.append(label)
        return image_names, labels

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        file_name = self.all_images[idx]
        
        if self.config['dataset']['type'] == 'txt':
            img = Image.open(file_name)
            try:
                if self.to_gray:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
            except Exception as e:
                print('Error image for {}'.format(file_name))
            
        elif self.config['dataset']['type'] == 'lmdb':
            image_key = self.all_images[idx]
            with self.env.begin(write=False) as txn:
                imgbuf = txn.get(image_key)
                buf = io.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    if self.to_gray:
                        img = Image.open(buf).convert('L')
                    else:
                        img = Image.open(buf).convert('RGB')
                except IOError:
                    print('Error Image for {}'.format(image_key))

        if self.transform is not None:
            img, width_ratio = self.transform(img)

        label = self.all_labels[idx]

        if not self.case_sensitive:
            label = label.lower()
        return img, label


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
            return None, None

        if self.training:
            images, labels = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return image_batch_tensor, labels


class Resize(object):
    def __init__(self, new_w, new_h, interpolation=Image.BILINEAR, gray_format=True):
        self.w, self.h = new_w, new_h
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.gray_format = gray_format

    def __call__(self, img):
        img_w, img_h = img.size
        if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:self.h, :] = img
                img = resize_img
                width = self.h
        elif img_w / img_h < self.w / self.h:
            ratio = img_h / self.h
            new_w = int(img_w / ratio)
            img = img.resize((new_w, self.h), self.interpolation)
            resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
            resize_img[0:self.h, 0:new_w, :] = img
            img = resize_img
            width = new_w
        else:
            img = img.resize((self.w, self.h), self.interpolation)
            resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
            resize_img[:, :, :] = img
            img = resize_img
            width = self.w

        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img, width / self.w

