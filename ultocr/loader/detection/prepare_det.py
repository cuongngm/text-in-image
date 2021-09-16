import os
import json
import cv2
import numpy as np


def labelme_to_det_format(json_dir, box_dir):
    for filename in os.listdir(json_dir):
        with open(os.path.join(json_dir, filename), 'r') as file:
            data = json.load(file)
            all_info = ''
            img_name = data['imagePath']
            boxes = data['shapes']
            for box in boxes:
                box_info = box['points']
                box_info = np.array(box_info).reshape(-1).tolist()
                box_info = list(map(int, box_info))
                box_info = list(map(str, box_info))
                if box['shape_type'] == 'polygon':
                    info = ','.join(box_info) + '\n'
                    all_info += info
                elif box['shape_type'] == 'rectangle':
                    info = box_info[0] + ',' + box_info[1] + ',' + box_info[2] + ',' + box_info[1] + ',' +\
                           box_info[2] + ',' + box_info[3] + ',' + box_info[0] + ',' + box_info[3] + '\n'
                    all_info += info
            with open(os.path.join(box_dir, img_name.replace('.jpg', '.txt')), 'w') as file_write:
                file_write.write(all_info)


def convert_labelme(img_dir, anns_dir, json_anns_dir):
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


if __name__ == '__main__':
    labelme_to_det_format('../../../dataset/json', '../../../dataset/box')