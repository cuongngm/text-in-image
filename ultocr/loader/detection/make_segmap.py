import cv2
import pyclipper
from shapely.geometry import Polygon
import numpy as np


class MakeSegMap:
    '''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    def __init__(self, algorithm='DB', min_text_size=5, shrink_ratio=0.4, is_training=True):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.is_training = is_training
        self.algorithm = algorithm

    def process(self, img, polys, ignore):
        '''
        img: [640, 640, 3]
        polys: list of array [14, 2] len N
        dontcare: list ignore len N
        '''
        h, w = img.shape[:2]
        # polys = [poly for poly in polys if Polygon(poly).buffer(0).is_valid]

        if self.is_training:
            polys, ignore = self.validate_polygons(
                polys, ignore, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polys)):
            poly = polys[i]
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])

            if ignore[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, poly.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore[i] = True
            else:
                polygon_shape = Polygon(poly)
                distance = polygon_shape.area * \
                           (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polys[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if len(shrinked) == 0:
                    cv2.fillPoly(mask, poly.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
        return img, gt, mask

    def validate_polygons(self, polygons, ignore, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore
        assert len(polygons) == len(ignore)
        # ignore_tags = [False] * len(polygons)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])
        return edge / 2.
