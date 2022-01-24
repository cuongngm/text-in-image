import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def sort_by_line(box_info, image):
    h, w = image.shape[:2]
    scale = max(h/800, 1)
    # box_info.sort(key=lambda x: compute_center(x)[1])
    all_same_row = []
    same_row = []
    for i in range(len(box_info)-1):
        # if compute_center(box_info[i+1])[1] - compute_center(box_info[i])[1] < 5 * scale:
        #     same_row.append(box_info[i])
        if box_info[i+1][0][1] - box_info[i][0][1] < 5:
            same_row.append(box_info[i])
        else:
            same_row.append(box_info[i])
            all_same_row.append(same_row)
            same_row = []
    same_row.append(box_info[-1])
    all_same_row.append(same_row)
    sort_same_row = []
    for same in all_same_row:
        same.sort(key=lambda x: x[1][0])
        sort_same_row.append(same)
    # print('len same row:', len(sort_same_row))
    return sort_same_row


def test_preprocess(img,
                    new_size=736,
                    pad=False):
    img = test_resize(img, size=new_size, pad=pad)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0)
    return img


def test_resize(img, size=736, pad=False):
    h, w, c = img.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)

    new_img = None
    if pad:
        new_img = np.zeros((size, size, c), img.dtype)
        new_img[:h, :w] = cv2.resize(img, (w, h))
    else:
        new_img = cv2.resize(img, (w, h))

    return new_img


def draw_bbox(img, result, color=(255, 0, 0), thickness=2):
    """
    :input: RGB img
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    h, w = img.shape[:2]
    for point in result:
        # point = point.astype(int)
        point = np.array(point).astype(int)
        point = point.reshape((-1, 1, 2))  
        cv2.polylines(img, [point], True, color, thickness)
    return img
