from typing import *
import math

import torch
import torch.nn as nn
from torchvision.ops import roi_pool, roi_align
from ultocr.model.common.resnet import resnet18, resnet50


class Encoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, ff_dim: int = 2048,
                 dropout: float = 0.1, num_layer: int = 6, max_len: int = 100,
                 image_features_dim: int = 512, image_encoder: str = 'resnet18',
                 roi_pooling_type: str = 'roi_pool', roi_pooling_size: Tuple = (7, 7)):
        super().__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layer)
        if image_encoder == 'resnet18':
            self.cnn = resnet18(image_features_dim)
        elif image_encoder == 'resnet50':
            self.cnn = resnet50(image_features_dim)
        else:
            raise NotImplementedError()
        assert roi_pooling_type in ['roi_pool', 'roi_align'], 'roi pooling type {} is not supported'.\
            format(roi_pooling_type)
        self.roi_pooling_type = roi_pooling_type
        self.roi_pooling_size = roi_pooling_size

        self.conv = nn.Conv2d(image_features_dim, d_model, kernel_size=roi_pooling_size)
        self.bn = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0 / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, images: torch.Tensor, boxes_coordinate: torch.Tensor,
                transcripts: torch.Tensor, src_key_padding_mask: torch.Tensor):
        """
        images: [B, C, H, W]
        boxes: [B, N, 8]
        transcripts: [B, N, T, D]
        src_key_padding_mask: []
        """
        b, n, t, d = transcripts.shape
        _, _, h_origin, w_origin = images.shape
        images = self.cnn(images)
        _, c, h, w = images.shape
        spatial_scale = h / h_origin
        rois_batch = torch.zeros(b, n, 5)  # 5 means [batch_idx, xmin, ymin, xmax, ymax]
        for i in range(b):
            doc_boxes = boxes_coordinate[i]
            pos = torch.stack([doc_boxes[:, 0], doc_boxes[:, 1], doc_boxes[:, 4], doc_boxes[:, 5]], dim=1)
            rois_batch[i, :, 1:5] = pos
            rois_batch[i, :, 0] = i
        if self.roi_pooling_type == 'roi_pool':
            images_segments = roi_pool(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        else:
            images_segments = roi_align(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        images_segments = self.relu(self.bn(self.conv(images_segments)))  # [B*N, D, 1, 1]
        images_segments = images_segments.squeeze()
        images_segments = images_segments.unsqueeze(1)  # [B*N, 1, D]

        transcripts_segments = transcripts + self.pe[:, :, :transcripts.size(2), :]
        transcripts_segments = transcripts_segments.reshape(b * n, t, d)
        transcripts_segments = self.transformer_encoder(transcripts_segments)

        images_segments = images_segments.expand_as(transcripts_segments)
        out = transcripts_segments + images_segments
        out = self.dropout(out)
        out = self.norm(out)
        return out