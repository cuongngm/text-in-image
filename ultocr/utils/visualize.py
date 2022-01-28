import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from ultocr.loader.detection.det_loader import DetLoader



def visualize_dbnet():
    with open('config/db_resnet50.yaml', 'r') as stream:
        cfg = yaml.safe_load(stream)

    train_dataset = DetLoader(cfg, is_training=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    # test_dataset = DetLoader(cfg, is_training=False)
    # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    # sample = next(iter(train_loader))

    samples = train_dataset[22]
    img = samples['img']
    gt = samples['gt']
    gt_mask = samples['gt_mask']
    thresh_map = samples['thresh_map']
    thresh_mask = samples['thresh_mask']
    fig = plt.figure(figsize=(10, 10))
    rows = 2
    columns = 2
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    # showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("First")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    # showing image
    plt.imshow(gt)
    plt.axis('off')
    plt.title("Second")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    # showing image
    plt.imshow(thresh_map)
    plt.axis('off')


def gen_color():
    """Generate BGR color schemes."""
    color_list = [(101, 67, 254), (154, 157, 252), (173, 205, 249),
                  (123, 151, 138), (187, 200, 178), (148, 137, 69),
                  (169, 200, 200), (155, 175, 131), (154, 194, 182),
                  (178, 190, 137), (140, 211, 222), (83, 156, 222)]
    return color_list


def draw_polygons(img, polys):
    """Draw polygons on image.
    Args:
        img (np.ndarray): The original image.
        polys (list[list[float]]): Detected polygons.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    dst_img = img.copy()
    color_list = gen_color()
    out_img = dst_img
    for idx, poly in enumerate(polys):
        poly = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(
            img,
            np.array([poly]),
            -1,
            color_list[idx % len(color_list)],
            thickness=cv2.FILLED)
        out_img = cv2.addWeighted(dst_img, 0.5, img, 0.5, 0)
    return out_img


def get_optimal_font_scale(text, width):
    """Get optimal font scale for cv2.putText.
    Args:
        text (str): Text in one box.
        width (int): The box width.
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale / 10,
            thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


def draw_texts(img, boxes, texts):
    """Draw boxes and texts on empty img.
    Args:
        img (np.ndarray): The original image.
        boxes (list[list[float]]): Detected bounding boxes.
        texts (list[str]): Recognized texts.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    color_list = gen_color()
    h, w = img.shape[:2]
    out_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
        Pts = np.array([new_box], np.int32)
        cv2.polylines(
            out_img, [Pts.reshape((-1, 1, 2))],
            True,
            color=color_list[idx % len(color_list)],
            thickness=1)
    out_img_pil = Image.fromarray(out_img)
    font_path = 'reg_module/utils/pala.ttf'
    draw = ImageDraw.Draw(out_img_pil)
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        min_x = int(min(box[0::2]))
        # max_y = int(np.mean(np.array(box[1::2])) + 0.2 * (max(box[1::2]) - min(box[1::2])))
        max_y = int(np.mean(np.array(box[1::2])))
        font_scale = get_optimal_font_scale(text, int(max(box[0::2]) - min(box[0::2])))
        # cv2.putText(out_img, text, (min_x, max_y), cv2.FONT_HERSHEY_SIMPLEX,
        #             font_scale, (0, 0, 0), 1)
        font = ImageFont.truetype(font_path, 15)
        draw.text((min_x, max_y), text, font=font, fill=(0, 0, 0, 0))
    out_img = np.array(out_img_pil)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return out_img


def det_recog_show_result(img, all_boxes, all_texts):
    """Draw `result`(boxes and texts) on `img`.
    Args:
        img (str or np.ndarray): The image to be displayed.
        end2end_res (dict): Text detect and recognize results.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box_vis_img = draw_polygons(img, all_boxes)
    text_vis_img = draw_texts(img, all_boxes, all_texts)

    h, w = img.shape[:2]
    out_img = np.ones((h, w * 2, 3), dtype=np.uint8)
    out_img[:, :w, :] = box_vis_img
    out_img[:, w:, :] = text_vis_img

    return out_img
