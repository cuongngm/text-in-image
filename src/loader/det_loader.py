import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.loader.augment import DetAugment
from src.loader.make_segmap import MakeSegMap
from src.loader.make_bordermap import MakeBorderMap
from src.utils.utils_function import resize


class DBLoader(Dataset):
    def __init__(self, config, img_dir, label_dir, is_training=True):
        super().__init__()
        self.crop_shape = config['base']['crop_shape']
        self.dataset_name = config['base']['dataset']
        self.img_list, self.label_list = self.get_base_info(img_dir, label_dir)
        self.is_training = is_training
        self.aug = DetAugment(self.crop_shape)
        self.MSM = MakeSegMap()
        self.MBM = MakeBorderMap()

    def get_base_info(self, img_dir, label_dir):
        img_list = []
        label_list = []
        print('Load with dataset {} type'.format(self.dataset_name))
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img_list.append(img_path)
            polys = []
            tags = []
            label_name = img_name.replace('.jpg', '.txt')
            label_name = 'gt_' + label_name
            with open(os.path.join(label_dir, label_name), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    poly = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                    if self.dataset_name == 'CTW1500':
                        poly = list(map(int, poly))
                        # poly = np.array(poly).reshape(-1, 2).tolist()
                        polys.append(poly)
                        tags.append(False)
                    elif self.dataset_name == 'ICDAR':
                        gt = poly[:-1]
                        if '#' in gt:
                            tags.append(True)
                        else:
                            tags.append(False)
                        poly = poly[:8]
                        poly = list(map(int, poly))
                        # poly = np.array(poly).reshape(-1, 2).tolist()
                        polys.append(poly)
            # label_list.append([np.array(polys), tags])
            label_list.append([polys, tags])
        assert len(img_list) == len(label_list), 'image with label not correct'
        return img_list, label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # index = self.img_list.index('dataset/CTW1500/train/img/CTW1500_0149.jpg')
        img_path = self.img_list[idx]
        polys, tags = self.label_list[idx]
        img = cv2.imread(img_path)
        if self.is_training:
            # augment
            img, polys = self.aug.random_scale(img, polys, self.crop_shape[0])
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
            data = {
                'img': img,
                'gt': gt,
                'gt_mask': gt_mask,
                'thresh_map': thresh_map,
                'thresh_mask': thresh_mask
            }
            return data
        else:
            img, polys, ignore = resize(img, polys, tags, size=self.crop_shape[0])
            data = {
                'img': img,
                'polys': polys,
                'ignore': ignore
            }
            return data


class DBTest(Dataset):
    def __init__(self, config):
        super(DBTest, self).__init__()
        self.img_list = self.get_img_files(config['val_load']['val_img_dir'])
        self.TSM = DetAugment(config['base']['crop_shape'])
        self.test_size = config['val_load']['test_size']
        self.config = config

    def get_img_files(self, img_dir):
        img_list = []
        for filename in os.listdir(img_dir):
            filepath = os.path.join(img_dir, filename)
            img_list.append(filepath)
        return img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        ori_img = cv2.imread(self.img_list[index])
        img = resize_image(ori_img, self.config['base']['algorithm'], self.test_size,
                           stride=self.config['val_load']['stride'])
        img = Image.fromarray(img).convert('RGB')
        img = self.TSM.normalize_img(img)
        return img, ori_img
