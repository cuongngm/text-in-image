import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from ultocr.loader.augment import DetAugment
from ultocr.loader.detection.make_segmap import MakeSegMap
from ultocr.loader.detection.make_bordermap import MakeBorderMap


class DetLoader(Dataset):
    def __init__(self, config, is_training=True):
        super().__init__()
        self.crop_shape = config['dataset']['crop_shape']
        self.dataset_type = config['dataset']['type']
        assert self.dataset_type in ['CTW1500', 'ICDAR'], 'data type is not correct'
        self.is_training = is_training

        if self.is_training:
            img_list, label_list = self.get_base_info(config['dataset']['train_load']['train_img_dir'],
                                                      config['dataset']['train_load']['train_label_dir'])
        else:
            img_list, label_list = self.get_base_info(config['dataset']['test_load']['test_img_dir'],
                                                      config['dataset']['test_load']['test_label_dir'])
        self.img_list, self.label_list = img_list, label_list
        # augment
        self.aug = DetAugment(self.crop_shape)
        self.MSM = MakeSegMap()
        self.MBM = MakeBorderMap()

    def get_base_info(self, img_dir, label_dir):
        img_list = []
        label_list = []
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img_list.append(img_path)
            polys = []
            ignore = []
            label_name = img_name.replace('.jpg', '.txt')
            # label_name = 'gt_' + label_name
            with open(os.path.join(label_dir, label_name), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    poly = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                    if self.dataset_type == 'CTW1500':
                        # x1, y1, x2, y2, ..., xn, yn
                        poly = list(map(int, poly))
                        polys.append(poly)
                        ignore.append(False)
                    elif self.dataset_type == 'ICDAR':
                        # x1, y1, x2, y2, x3, y3, x4, y4, transcripts
                        poly = poly[:8]
                        poly = list(map(int, poly))
                        polys.append(poly)
                        ignore.append(False)
            # label_list.append([np.array(polys), tags])
            label_list.append([polys, ignore])
        assert len(img_list) == len(label_list), 'image with label not correct'
        return img_list, label_list

    def test_resize(self, img, polys, pad=False):
        h, w, c = img.shape
        new_size = self.crop_shape[0]
        scale_h = new_size / h
        scale_w = new_size / w
        scale = min(scale_w, scale_h)
        new_h = int(scale * h)
        new_w = int(scale * w)
        if pad:
            new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
            new_img[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))
        else:
            new_img = cv2.resize(img, (new_w, new_h))
        new_polys = []
        for poly in polys:
            poly = np.array(poly).astype(np.float64)
            poly = poly * scale
            new_polys.append(poly)
        return new_img, new_polys

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        polys = self.label_list[idx][0]
        ignore = self.label_list[idx][1]
        img = cv2.imread(img_path)
        if self.is_training:
            # augment
            img, polys = self.aug.random_scale(img, polys, self.crop_shape[0])
            img, polys = self.aug.random_rotate(img, polys)
            img, polys = self.aug.random_flip(img, polys)
            img, polys, ignore = self.aug.random_crop_db(img, polys, ignore)
            # make segment map, make border map
            img, gt, gt_mask = self.MSM.process(img, polys, ignore)
            img, thresh_map, thresh_mask = self.MBM.process(img, polys, ignore)

            img = Image.fromarray(img).convert('RGB')
            img = transforms.ColorJitter(brightness=32.0/255, saturation=0.5)(img)
            img = self.aug.normalize_img(img)

            gt = torch.from_numpy(gt).float()
            gt_mask = torch.from_numpy(gt_mask).float()
            thresh_map = torch.from_numpy(thresh_map).float()
            thresh_mask = torch.from_numpy(thresh_mask).float()

            data = {
                'img': img,
                'gt': gt,
                'gt_mask': gt_mask,
                'thresh_map': thresh_map,
                'thresh_mask': thresh_mask
            }
        else:
            img, polys = self.test_resize(img, polys, pad=True)
            img = Image.fromarray(img).convert('RGB')
            img = self.aug.normalize_img(img)
            data = {
                'img': img,
                'polys': polys,
            }
        return data
