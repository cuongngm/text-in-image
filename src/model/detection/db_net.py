import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utils_function import create_module


class DBNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = create_module(config['model']['common'])(config['base']['pretrained'])
        self.head = create_module(config['model']['head'])(config['base']['in_channels'],
                                                           config['base']['inner_channels'])
        self.seg_out = create_module(config['model']['segout'])(config['base']['inner_channels'],
                                                                config['base']['k'],
                                                                config['base']['adaptive'])

    def forward(self, x):
        """
                :return: Train mode: prob_map, threshold_map, appro_binary_map
                :return: Eval mode: prob_map, threshold_map
                """
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_body_out = self.head(backbone_out)
        segmentation_head_out = self.seg_out(segmentation_body_out)
        y = F.interpolate(segmentation_head_out, size=(H, W),
                          mode='bilinear', align_corners=True)
        return y


class DetLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss = create_module(config['loss']['function'])(config['loss']['l1_scale'],
                                                              config['loss']['bce_scale'])

    def forward(self, pre_batch, gt_batch):
        return self.loss(pre_batch, gt_batch)
