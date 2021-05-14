import torch
import torch.nn as nn
from src.utils.utils_function import create_module


class DBNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = create_module(config['model']['backbone'])(config['base']['pretrained'])
        self.head = create_module(config['model']['head'])(config['base']['in_channels'],
                                                           config['base']['inner_channels'])
        self.seg_out = create_module(config['model']['segout'])(config['base']['inner_channels'],
                                                                config['base']['k'],
                                                                config['base']['adaptive'])

    def forward(self, data):
        if self.training:
            img, gt, gt_mask, thresh, thresh_mask = data
            if torch.cuda.is_available():
                img, gt, gt_mask, thresh, thresh_mask = img.cuda(), gt.cuda(), gt_mask.cuda(),\
                                                        thresh.cuda(), thresh_mask.cuda()
            gt_batch = dict(gt=gt)  # = gt_batch['gt'] = gt
            gt_batch['gt_mask'] = gt_mask
            gt_batch['thresh'] = thresh
            gt_batch['thresh_mask'] = thresh_mask
        else:
            img = data
        x = self.backbone(img)
        x = self.head(x)
        x = self.seg_out(x, img)
        if self.training:
            return x, gt_batch
        return x


class DBLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss = create_module(config['loss']['function'])(config['loss']['l1_scale'],
                                                              config['loss']['bce_scale'])

    def forward(self, pre_batch, gt_batch):
        return self.loss(pre_batch, gt_batch)
