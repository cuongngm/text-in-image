import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.loader.augment import DetAugment
from src.loader.make_segmap import MakeSegMap
from src.loader.make_bordermap import MakeBorderMap


class DBLoaderTrain(Dataset):
    def __init__(self, config):
        super().__init__()
        self.crop_shape = config['base']['crop_shape']
        self.img_list, self.label_list = self.get_base_info(config['train_load']['train_img_dir'],
                                                            config['train_load']['train_label_dir'])
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
            tags = []
            label_name = img_name.replace('.jpg', '.txt')
            with open(os.path.join(label_dir, label_name), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    item = {}
                    poly = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                    poly = list(map(int, poly))
                    poly = np.array(poly).reshape(-1, 2).tolist()
                    polys.append(poly)
                    tags.append('ignore')
            label_list.append([np.array(polys), tags])
        assert len(img_list) == len(label_list), 'image with label not correct'
        return img_list, label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        polys, tags = self.label_list[idx]
        img = cv2.imread(img_path)
        # augment
        img, polys = self.aug.random_rotate(img, polys)
        img, polys = self.aug.random_flip(img, polys)
        img, polys, ignore = self.aug.random_crop_db(img, polys, ignore=tags)
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
        return img, gt, gt_mask, thresh_map, thresh_mask
