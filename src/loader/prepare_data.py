import os
import cv2
import json
import numpy as np
from shutil import copyfile
from random import shuffle
import pandas as pd


def crop_poly(img, points):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array(points)
    points = points.reshape(1, -1, 2)
    cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(points)
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped


def split(gt_path):
    with open(gt_path, 'r') as file:
        lines = file.readlines()
        num = len(lines)
        index = np.arange(num)
        shuffle(index)
        index_train = index[0: int(len(index) * 0.8)]
        index_val = index[int(len(index) * 0.8):]
        train_data = []
        val_data = []
        for idx in index_train:
            train_data.append(lines[idx])
        for idx in index_val:
            val_data.append(lines[idx])
        with open('../../dataset/vietnamese/train.txt', 'w') as f_train:
            for data in train_data:
                f_train.write(data)
        with open('../../dataset/vietnamese/test.txt', 'w') as f_test:
            for data in val_data:
                f_test.write(data)


class BillKIE:
    def __init__(self, img_dir, box_dir, key_dir):
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.key_dir = key_dir

    def rename(self):
        for key_name in os.listdir(self.key_dir):
            os.rename(os.path.join(self.key_dir, key_name),
                      os.path.join(self.key_dir, key_name.replace('key_', '')))

    def copy(self, box_and_tran='../../dataset/bill2/boxes'):
        for box_name in os.listdir(self.box_dir):
            copyfile(os.path.join(self.box_dir, box_name),
                     os.path.join(box_and_tran, box_name.replace('.txt', '.tsv')))

    def key(self):
        for key_name in os.listdir(self.key_dir):
            key_path = os.path.join(self.key_dir, key_name)
            list_entity = dict()
            with open(key_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    entity = line.split('\t')
                    if len(entity) == 2:
                        entity_key = entity[0]
                        entity_value = entity[1]
                    else:
                        entity_key = entity[0]
                        entity_value = ''
                    list_entity[entity_key] = entity_value
            with open(os.path.join('../../dataset/bill2/entities', key_name), 'w') as file_write:
                file_write.write(json.dumps(list_entity))

    def write_sample_list(self):
        img_list = os.listdir(self.img_dir)
        # img_name = list(img.replace('.jpg', '') for img in img_list)
        img_type = ['receipt'] * len(img_list)
        dic = {'type': img_type, 'img_list': img_list}
        df = pd.DataFrame(dic)
        df.to_csv('../../dataset/bill2/test/test_samples_list.csv', header=False)

    def new_box(self, new_box='../../dataset/bill2/new_box'):
        for box_name in os.listdir(self.box_dir):
            list_old_box = []
            list_trans = []
            news = ''
            with open(os.path.join(new_box, box_name), 'r') as file:
                new_lines = file.readlines()
            with open(os.path.join(self.box_dir, box_name), 'r') as gt_file:
                lines = gt_file.readlines()
                for line in lines:
                    line = line.strip().split(',')
                    old_box = ','.join(line[:8])
                    trans = ','.join(line[8:])
                    list_old_box.append(old_box)
                    list_trans.append(trans)
            # print('new_lines', new_lines)
            # print('old', list_old_box)
            # print('trans', list_trans)
            for box, tran in zip(new_lines, list_trans):
                new = box[:-1] + ',' + tran + '\n'
                news += new
            with open(os.path.join('../../dataset/bill2/new', box_name), 'w') as file:
                file.write(news)


class VietnameseDataset:
    def __init__(self, img_dir, label_dir):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir

    def rename(self):
        for filename in os.listdir(self.img_dir):
            idx_name = filename[2:6]
            idx_name = int(idx_name)
            os.rename(os.path.join(self.img_dir, filename), os.path.join(self.img_dir, str(idx_name) + '.jpg'))

    def visualize(self):
        for filename in os.listdir(self.img_dir):
            img_path = os.path.join(self.img_dir, filename)
            gt_path = 'gt_' + filename.replace('.jpg', '.txt')
            img = cv2.imread(img_path)
            with open(os.path.join(self.label_dir, gt_path), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    poly = line.strip().split(',')
                    poly = poly[:8]
                    poly = list(map(int, poly))
                    cv2.rectangle(img, (poly[0], poly[1]), (poly[4], poly[5]), (0, 255, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            break

    def crop(self):
        all_text = ''
        for idx_img, filename in enumerate(os.listdir(self.img_dir)):
            img_path = os.path.join(self.img_dir, filename)
            gt_path = 'gt_' + filename.replace('.jpg', '.txt')
            img = cv2.imread(img_path)
            with open(os.path.join(self.label_dir, gt_path), 'r') as file:
                lines = file.readlines()
                for idx, line in enumerate(lines):
                    poly = line.strip().split(',')
                    label = poly[8:]
                    label = ','.join(label)
                    if '#' in label:
                        continue
                    poly = poly[:8]
                    # label = ','.join(label)
                    poly = list(map(int, poly))
                    crop_img = crop_poly(img, poly)
                    try:
                        cv2.imwrite('../../dataset/vietnamese/word_img/val_{}_{}.jpg'.format(idx_img, idx), crop_img)
                        all_text += 'word_img/val_{}_{}.jpg'.format(idx_img, idx) + '\t' + label + '\n'
                    except:
                        print(os.path.join(self.label_dir, gt_path))
        with open('../../dataset/vietnamese/val.txt', 'w') as file:
            file.write(all_text)


class BillDataset:
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

    def move(self):
        for filename in os.listdir(self.img_dir):
            if filename.endswith('.jpg'):
                os.rename(os.path.join(self.img_dir, filename), os.path.join('../../dataset/dataHoaDon/hoadon', filename))

    def split(self):
        train_data = []
        print(len(os.listdir(self.label_dir)))
        for filename in os.listdir(self.label_dir):
            with open(os.path.join(self.label_dir, filename), 'r') as file:
                text = file.read().strip()
                data = 'hoadon/' + filename.replace('.txt', '.jpg') + '\t' + text + '\n'
                train_data.append(data)
        print(len(train_data))
        with open('../../dataset/dataHoaDon/test.txt', 'w') as f_train:
            for data in train_data:
                f_train.write(data)



if __name__ == '__main__':
    dataset = BillKIE(img_dir='../../dataset/bill2/test/images',
                      box_dir='../../dataset/bill2/test/boxes_and_transcripts',
                      key_dir='../../dataset/bill2/test/entities')
    dataset.write_sample_list()
