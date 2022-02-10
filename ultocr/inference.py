import cv2
import yaml
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from ultocr.utils.download import download_weights
from ultocr.utils.utils_function import create_module
from ultocr.loader.recognition.reg_loader import TextInference
from ultocr.utils.det_utils import four_point_transform, sort_by_line, test_preprocess, draw_bbox
from ultocr.loader.recognition.translate import LabelConverter
from ultocr.loader.recognition.reg_loader import Resize
from ultocr.model.recognition.postprocess import greedy_decode_with_probability
import mlflow
from mlflow.models.signature import infer_signature


class Detection:
    def __init__(self, weight, cfg):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = create_module(cfg['model']['function'])(cfg)
        state_dict = torch.load(weight, map_location=self.device)['model_state_dict']
        # state_dict = change_state_dict(state_dict)
        model.load_state_dict(state_dict)
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
        boxes_list, scores_list = self.seg_obj(batch, preds, inference=True)
 
        boxes_list, scores_list = boxes_list[0].tolist(), scores_list[0]
        
        # img_rs = draw_bbox(img, np.array(boxes_list), color=(0, 0, 255), thickness=1)
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
            rs = draw_bbox(img, np.array(after_sort), color=(0, 0, 255), thickness=2)
            # return rs
            all_warped = []
            for index, boxes in enumerate(after_sort2):
                warped = four_point_transform(img, boxes)
                # warped = crop_box(img, boxes)
                all_warped.append(warped)

            det_result['img'] = rs
            det_result['box_coordinate'] = after_sort
            det_result['boundary_result'] = all_warped
            return det_result


class Recognition:
    def __init__(self, weight, cfg):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.img_w = cfg['dataset']['new_shape'][0]
        self.img_h = cfg['dataset']['new_shape'][1]
        self.batch = 16
        vocab_file = download_weights('1Lo9L_k63M7vpiR10zii5nzL4GGUSuntM')
        self.convert = LabelConverter(classes=vocab_file, max_length=100, ignore_over=False)
        model = create_module(cfg['model']['function'])(cfg)
        state_dict = torch.load(weight, map_location=self.device)['model_state_dict']
        model.load_state_dict(state_dict)
        # model = mlflow.pytorch.load_model(model_uri='abc')
        self.model = model.to(self.device)
        self.model.eval()  

    def recognize(self, list_img):
        # new_list_img = [Image.fromarray(img) for img in list_img]
        text_dataset = TextInference(list_img, transform=Resize(self.img_w, self.img_h, gray_format=False))
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
        """
        signature = infer_signature(model_input=images.detach().cpu().numpy(),
                                    model_output=outputs.detach().cpu().numpy())
        mlflow.pytorch.save_model(self.model, 'abc', signature=signature)
        """
        return pred_results


class OCR:
    def __init__(self, det_model='DB', reg_model='MASTER',
                 det_config='1ca-ym1bAZTmgPyEL78PRnJ-_Jn1VPT-C', reg_config='1xL_DWV9Yc5qwc9ucVHlYv-xyrrvkOzzL',
                 det_weight='1KWKMiN5iRDtqb1l3FO3o1z6ThxLvfq9a', reg_weight='1V9CGvqC_SsXOEXiNGlRbZxp9fn0qH6Lf'):
        assert det_model in ['DB'], '{} model is not implement'.format(det_model)
        assert reg_model in ['MASTER'], '{} model is not implement'.format(reg_model)
        if '.yaml' not in det_config:
            det_config = download_weights(det_config)
        with open(det_config, 'r') as stream:
            det_cfg = yaml.safe_load(stream)
        if '.pth' not in det_weight:
            det_weight = download_weights(det_weight)
        self.detection = Detection(det_weight, det_cfg)

        if '.yaml' not in reg_config:
            reg_config = download_weights(reg_config)
        with open(reg_config, 'r') as stream:
            reg_cfg = yaml.safe_load(stream)
        if '.pth' not in reg_weight:
            reg_weight = download_weights(reg_weight)
        self.recognition = Recognition(reg_weight, reg_cfg)
          
    def get_result(self, input_image):
        # input_image: PIL image
        img = np.array(input_image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        det_result = self.detection.detect(img)
        img_rs = det_result['img']
        all_img_crop = det_result['boundary_result']
        if len(all_img_crop) == 0:
            result = 'khong co text trong anh'
            return result
        all_img_pil = []
        for idx, img_crop in enumerate(all_img_crop):
            img_pil = Image.fromarray(img_crop.astype('uint8'), 'RGB')
            all_img_pil.append(img_pil)
        result = self.recognition.recognize(all_img_pil)
        return result

