import os
import json
import cv2
import numpy as np
from tqdm import tqdm


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
    # labelme_to_det_format(json_dir='../dataset/layoutlm/json', box_dir='../dataset/layoutlm/box')
    crop_img(img_dir='../../../dataset/layoutlm/img', box_dir='../../../dataset/layoutlm/box',
             save_dir='../../../dataset/layoutlm/crop_img')