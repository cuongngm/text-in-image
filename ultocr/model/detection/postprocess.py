import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper


# postprocessing
class DBPostProcess:
    def __init__(self, config):
        self.thresh = config['post_process']['thresh']
        self.box_thresh = config['post_process']['box_thresh']
        self.max_candidates = config['post_process']['max_candidates']
        self.is_poly = config['post_process']['is_poly']
        self.unclip_ratio = config['post_process']['unclip_ratio']
        self.min_size = config['post_process']['min_size']

    def __call__(self, batch, pred, is_output_polygon=False):
        """
        :param batch: (image, polygons, ignore_Tags)
        image: tensor (N, C, H, W) (1, 3, 640, 640)
        polygons: tensor (N, K, 4, 2)
        ignore_tags: tensor (N, K)
        :param pred: binary, thresh, thresh_binary
        binary: text region segmentation map, with shape (N, H, W)
        thresh: thresh hold prediction (N, H, W)
        thresh_binary: binarized with threshold (N, H, W)
        """
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.size(0)):
            height, width = batch['shape'][batch_index]
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index],
                                                          segmentation[batch_index],
                                                          width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index],
                                                       segmentation[batch_index],
                                                       width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (H, W) whose values are binarized as {0, 1}
        """
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()
        pred = pred.cpu().numpy()
        height, width = bitmap.shape[:2]
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue
            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0,
                                dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0,
                                dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def get_mini_boxes(self, contour):
        try:
            bounding_box = cv2.minAreaRect(contour)
            points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
            index_1, index_2, index_3, index_4 = 0, 1, 2, 3
            if points[1][1] > points[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if points[3][1] > points[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2
            box = [points[index_1], points[index_2], points[index_3], points[index_4]]
            return box, min(bounding_box[1])
        except Exception:
            return [], -1

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        x_min = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        x_max = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        y_min = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        y_max = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        box[:, 0] -= x_min
        box[:, 1] -= y_min
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[y_min: y_max + 1, x_min:x_max + 1], mask)[0]
