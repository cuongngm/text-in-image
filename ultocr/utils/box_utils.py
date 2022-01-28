"""
merge box and ocr result in same line (word level methods)
"""
import numpy as np
from shapely.geometry import Polygon, MultiPoint


def cal_iou(bbox1, bbox2):
    bbox1 = np.array(bbox1).reshape(-1, 2)
    bbox2 = np.array(bbox2).reshape(-1, 2)
    bbox1_poly = Polygon(bbox1).convex_hull
    bbox2_poly = Polygon(bbox2).convex_hull
    union_poly = np.concatenate((bbox1, bbox2))
    if not bbox1_poly.intersects(bbox2_poly):
        iou = 0
    else:
        inter_area = bbox1_poly.intersection(bbox2_poly).area
        union_area = MultiPoint(union_poly).convex_hull.area
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / bbox2_poly.area
    return iou


def merge_word2line(boxes_line, boxes_word, texts):
    match_pair_list = []
    not_match = []
    for idx, (box_word, text) in enumerate(zip(boxes_word, texts)):
        max_iou = 0
        max_match = [None, None]
        for j, box_line in enumerate(boxes_line):
            iou = cal_iou(box_line, box_word)
            if iou > max_iou:
                max_match[0], max_match[1] = idx, j
                max_iou = iou
        if max_match[0] is None:
            not_match.append(idx)
            continue
        match_pair_list.append(max_match)
    # print('match_pair_list', match_pair_list)
    # print(not_match) 
    match_pair_dict = dict()
    for match_pair in match_pair_list:
        if match_pair[1] not in match_pair_dict.keys():
            match_pair_dict[match_pair[1]] = [match_pair[0]]
            if match_pair[0] - 1 in not_match:
                match_pair_dict[match_pair[1]].append(match_pair[0] - 1)
        else:
            match_pair_dict[match_pair[1]].append(match_pair[0])
            if match_pair[0] - 1 in not_match:
                match_pair_dict[match_pair[1]].append(match_pair[0] - 1)
    
    # print(match_pair_dict)

    words_arrange = []
    boxes_arrange = []
    for k in match_pair_dict.keys():
        idx_same_line = match_pair_dict[k]
        word_in_line = []
        box_in_line = []
        for idx in idx_same_line:
            box_in_line.append(boxes_word[idx])
        box_in_line.sort(key=lambda x:x[0])
        for box in box_in_line:
            words_arrange.append(texts[boxes_word.index(box)])
            boxes_arrange.append(box)
    return words_arrange, boxes_arrange


def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def stitch_boxes_into_lines(boxes, max_x_dist=10, min_y_overlap_ratio=0.8):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            merged_box['text'] = ' '.join(
                [x_sorted_boxes[idx]['text'] for idx in box_group])
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            for idx in box_group:
                x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)
            merged_box['box'] = [
                x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
            ]
            merged_boxes.append(merged_box)
    return merged_boxes
