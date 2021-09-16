import torch.nn as nn
import torch.nn.functional as F
from ultocr.utils.utils_function import create_module


class DBNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = create_module(config['model']['backbone']['function'])(config['model']['backbone']['pretrained'])
        self.head = create_module(config['model']['head']['function'])(config['model']['head']['in_channels'],
                                                                       config['model']['head']['inner_channels'])
        self.seg_out = create_module(config['model']['segout']['function'])(config['model']['segout']['inner_channels'],
                                                                            config['model']['segout']['k'],
                                                                            config['model']['segout']['adaptive'])

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
