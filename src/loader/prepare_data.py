import os
import cv2
import numpy as np


def crop_four_corner(directory):
    for filename in os.listdir(directory):
        img_path = os.path.join('img', filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        with open(os.path.join(directory, filename), 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                box = line.strip().split(',')
                tl = [box[0], box[1]]
                tr = [box[2], box[3]]
                br = [box[4], box[5]]
                bl = [box[6], box[7]]
                pst1 = np.float32([tl, tr, br, bl])
                pst2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                M = cv2.getPerspectiveTransform(pst1, pst2)
                dst = cv2.warpPerspective(img, M, (w, h))
                cv2.imwrite('../../dataset/custom_data/{}_{}.jpg'.format(filename, idx), dst)


def crop_two_corner(directory):
    for filename in os.listdir(directory):
        img_path = os.path.join('img', filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        with open(os.path.join(directory, filename), 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().split(',')
                box = line[:8]
                box = [int(b) for b in box]
                transcript = line[8:]
                transcript = ','.join(transcript)
                box_crop = img[box[1]:box[5], box[0]:box[4]]
                cv2.imwrite('../../dataset/custom_data/{}_{}.jpg'.format(filename, idx), box_crop)
