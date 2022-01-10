import cv2
import argparse
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from ultocr.utils.utils_function import create_module
from ultocr.loader.recognition.reg_loader import TextInference
from ultocr.utils.det_utils import four_point_transform, sort_by_line, test_preprocess, draw_bbox
from ultocr.utils.reg_utils import ResizeWeight, ConvertLabelToMASTER, greedy_decode_with_probability


class Detection:
    def __init__(self, cfg):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = create_module(cfg['model']['function'])(cfg)
        model.load_state_dict(torch.load('saved/ckpt/DBnet/0108_112716/last_cp.pth', map_location=self.device)['state_dict'])
        self.model = model.to(self.device)
        self.model.eval()
        self.seg_obj = create_module(cfg['post_process']['function'])(cfg)

    def detect(self, img):
        det_result = {}
        h_origin, w_origin = img.shape[:2]
        tmp_img = test_preprocess(img, new_size=736, pad=False).to(self.device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            preds = self.model(tmp_img)
        batch = {'shape': [(h_origin, w_origin)]}
        boxes_list, scores_list = self.seg_obj(batch, preds, is_output_polygon=False)
        boxes_list, scores_list = boxes_list[0].tolist(), scores_list[0]
        img_rs = draw_bbox(img, np.array(boxes_list), color=(0, 0, 255), thickness=1)
        boxes_list.sort(key=lambda x: x[0][1])
        boxes_list_remove = []
        for boxes in boxes_list:
            if boxes[0] == boxes[2] or boxes[1] == boxes[3]:
                continue
            else:
                boxes_list_remove.append(boxes)
        if len(boxes_list_remove) == 0:
            det_result['img'] = img
            det_result['box_coordinate'] = []
            det_result['boundary_result'] = []
            return det_result
        else:
            sort_box_list = sort_by_line(boxes_list_remove, img)
            after_sort = []
            after_sort2 = []
            for same_row in sort_box_list:
                for box in same_row:
                    point = np.array(box)
                    point = point.astype(int)
                    after_sort2.append(point)

                    box = np.array(box).reshape(-1).tolist()
                    after_sort.append(box)
            # rs = draw_bbox(img, np.array(after_sort), color=(0, 0, 255), thickness=2)
            # return rs
            all_warped = []
            for index, boxes in enumerate(after_sort2):
                warped = four_point_transform(img, boxes)
                # warped = crop_box(img, boxes)
                all_warped.append(warped)

            det_result['img'] = img_rs
            det_result['box_coordinate'] = after_sort
            det_result['boundary_result'] = all_warped
            return det_result


class Recognition:
    def __init__(self, cfg):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.img_w = cfg['dataset']['img_w']
        self.img_h = cfg['dataset']['img_h']
        self.batch = 16
        self.convert = ConvertLabelToMASTER(vocab_file='ultocr/utils/vocab.txt',
                                            max_length=100, ignore_over=False)
        model = create_module(cfg['functional']['master'])(cfg)
        state_dict = torch.load('saved/weight/master_30e.pth', map_location=self.device)
        model.load_state_dict(state_dict['model_state_dict'])
        self.model = model.to(self.device)
        self.model.eval()

    def recognize(self, list_img):
        text_dataset = TextInference(list_img, transform=ResizeWeight((self.img_w, self.img_h), gray_format=False))
        text_loader = DataLoader(text_dataset, batch_size=self.batch, shuffle=False, num_workers=4, drop_last=False)
        pred_results = []
        for step_idx, data_item in enumerate(text_loader):
            # print('data_item', data_item.size())
            images = data_item
            with torch.no_grad():
                images = images.to(self.device)
                # if hasattr(self.model, 'module'):
                #     model = self.model.module
                outputs, probs = greedy_decode_with_probability(self.model, images, self.convert.max_length,
                                                                self.convert.SOS,
                                                                padding_symbol=self.convert.PAD,
                                                                device=self.device, padding=True)
            for index, (pred, prob) in enumerate(zip(outputs[:, 1:], probs)):
                pred_text = ''
                pred_score_list = []
                for i in range(len(pred)):
                    if pred[i] == self.convert.EOS:
                        pred_score_list.append(prob[i])
                        break
                    if pred[i] == self.convert.UNK:
                        continue

                    decoder_char = self.convert.decode(pred[i])
                    pred_text += decoder_char
                    pred_score_list.append(prob[i])
                pred_score = sum(pred_score_list) / len(pred_score_list)
                # pred_item = {'result': pred_text,
                #              'prob': pred_score}
                # if pred_score > 0.5:
                pred_results.append(pred_text)
                # else:
                #     pred_results.append('')
        return pred_results


class End2end:
    def __init__(self, img_path, det_model='DB', reg_model=None,
                 det_config='config/db_resnet50.yaml', reg_config='config/master.yaml'):
        self.img_path = img_path
        assert det_model in ['DB'], '{} model is not implement'.format(det_model)
        # assert reg_model in ['MASTER'], '{} model is not implement'.format(reg_model)
        with open(det_config, 'r') as stream:
            det_cfg = yaml.safe_load(stream)
        self.detection = Detection(det_cfg)
        if reg_model is not None: 
            with open(reg_config, 'r') as stream:
                reg_cfg = yaml.safe_load(stream)
            self.recognition = Recognition(reg_cfg)
          
    def get_result(self):
        img = cv2.imread(self.img_path)
        det_result = self.detection.detect(img)
        """
        all_img_crop = det_result['boundary_result']
        if len(all_img_crop) == 0:
            result = 'khong co text trong anh'
            return result
        all_img_pil = []
        for idx, img_crop in enumerate(all_img_crop):
            img_pil = Image.fromarray(img_crop.astype('uint8'), 'RGB')
            all_img_pil.append(img_pil)
        result = self.recognition.recognize(all_img_pil)
        """
        return det_result

