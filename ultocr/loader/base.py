import cv2
import os
import math
import json
import numpy as np
from tqdm import tqdm
from random import shuffle
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistValSampler(Sampler):
    # DistValSampler distributes batches equally (based on batch size) to every gpu (even if there aren't enough samples)
    # This instance is used as batch_sampler args of validation dtataloader,
    # to guarantee every gpu validate different samples simultaneously
    # WARNING: Some baches will contain an empty array to signify there aren't enough samples
    # distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = 0
            self.world_size = 1

        # expected number of batches per process. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_steps = math.ceil(len(self.indices) / self.world_size / self.batch_size)

        # num_samples = total samples / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_steps * self.batch_size

    def __iter__(self):
        current_rank_offset = self.num_samples * self.global_rank
        current_sampled_indices = self.indices[
                                  current_rank_offset:min(current_rank_offset + self.num_samples, len(self.indices))]

        for step in range(self.expected_num_steps):
            step_offset = step * self.batch_size
            # yield sampled_indices[offset:offset + self.batch_size]
            yield current_sampled_indices[step_offset:min(step_offset + self.batch_size, len(current_sampled_indices))]

    def __len__(self):
        return self.expected_num_steps

    def set_epoch(self, epoch):
        return
    
    
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


def labelme_to_det_format(json_dir, box_dir):
    """
    convert data from labelme format to detection format
    :param json_dir:
    :param box_dir:
    :return:
    """
    for filename in os.listdir(json_dir):
        with open(os.path.join(json_dir, filename), 'r') as file:
            data = json.load(file)
            all_info = ''
            img_name = data['imagePath']
            boxes = data['shapes']
            for box in boxes:
                entity = box['label']
                box_info = box['points']
                box_info = np.array(box_info).reshape(-1).tolist()
                box_info = list(map(int, box_info))
                box_info = list(map(str, box_info))
                if box['shape_type'] == 'polygon':
                    info = ','.join(box_info) + ',' + entity + '\n'
                    all_info += info
                elif box['shape_type'] == 'rectangle':
                    info = box_info[0] + ',' + box_info[1] + ',' + box_info[2] + ',' + box_info[1] + ',' +\
                           box_info[2] + ',' + box_info[3] + ',' + box_info[0] + ',' + box_info[3] + ',' + entity + '\n'
                    all_info += info
            with open(os.path.join(box_dir, img_name.replace('.jpg', '.txt')), 'w') as file_write:
                file_write.write(all_info)


def det_format_to_labelme(img_dir, anns_dir, json_anns_dir):
    """
    convert detection format to labelme format
    :param img_dir:
    :param anns_dir:
    :param json_anns_dir:
    :return:
    """
    dic = {'version': '4.5.6', 'flags': {}, 'imageData': None}
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        anns_name = img_name.replace('.jpg', '.txt')
        anns_path = os.path.join(anns_dir, anns_name)
        dic['imagePath'] = img_name
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        dic['imageHeight'] = h
        dic['imageWeight'] = w
        list_shapes = []
        with open(anns_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                dic_shapes = {}
                line = line.strip().split(',')
                assert len(line) == 8, print(anns_path)
                line = [int(li) for li in line]
                points = [[line[0], line[1]], [line[2], line[3]], [line[4], line[5]], [line[6], line[7]]]
                dic_shapes['label'] = 'text'
                dic_shapes['points'] = points
                dic_shapes['group_id'] = None
                dic_shapes['shape_type'] = 'polygon'
                dic_shapes['flags'] = {}
                list_shapes.append(dic_shapes)
        dic['shapes'] = list_shapes
        with open(os.path.join(json_anns_dir, img_name.replace('.jpg', '.json')), 'w') as out_file:
            json.dump(dic, out_file)


def crop_img(img_dir, box_dir, save_dir):
    """
    crop box image from coordinate
    :param img_dir:
    :param box_dir:
    :return:
    """
    for filename in tqdm(sorted(os.listdir(img_dir))):
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        # h, w = img.shape[:2]
        box_path = os.path.join(box_dir, filename.replace('.jpg', '.txt'))
        with open(box_path, 'r') as fr:
            lines = fr.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().split(',')
                box = line[:8]
                box = list(map(int, box))
                box = np.array(box).reshape(-1, 2).tolist()
                tl, bl, tr, br = box[0], box[3], box[1], box[2]
                w = max((tr[0] - tl[0]), (br[0] - bl[0]))
                h = max((bl[1] - tl[1]), (br[1] - tr[1]))
                pts1 = np.float32([tl, tr, bl, br])
                pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(img, M, (w, h))
                cv2.imwrite(os.path.join(save_dir, filename[:-4] + '_' + str(idx) + '.jpg'), dst)


if __name__ == '__main__':
    labelme_to_det_format(json_dir='../../../dataset/CDP/json', box_dir='../../../dataset/CDP/box')
    # crop_img(img_dir='../../../dataset/layoutlm/img', box_dir='../../../dataset/layoutlm/box',
    #          save_dir='../../../dataset/layoutlm/crop_img')
