import numpy as np
import cv2
import torch


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
                    mean=[103.939, 116.779, 123.68],
                    to_tensor=True,
                    pad=False):
    img = test_resize(img, size=736, pad=pad)

    img = img.astype(np.float32)
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img = np.expand_dims(img, axis=0)

    if to_tensor:
        img = torch.Tensor(img.transpose(0, 3, 1, 2))

    return img


def test_resize(img, size=640, pad=False):
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