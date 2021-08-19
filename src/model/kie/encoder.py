from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.ops import roi_pool, roi_align
from src.model.backbone.resnet import resnet18, resnet50


class Encoder(nn.Module):
    def __init__(self, out_dim: int, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.1, num_layer: int = 6,
                 image_encoder: str = 'resnet18', image_feature_dim: int = 512, max_len: int = 100,
                 roi_pooling_size: Tuple[int, int] = (7, 7), roi_pooling_mode='roi_pool'):
        super().__init__()
        self.roi_pooling_mode = roi_pooling_mode
        assert roi_pooling_mode in ['roi_pool', 'roi_align'], 'roi pooling mode: {} is not supported'.format(roi_pooling_mode)
        self.roi_pooling_size = tuple(roi_pooling_size)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layer)
        if image_encoder == 'resnet18':
            self.cnn = resnet18(output_channels=image_feature_dim)
        elif image_encoder == 'resnet50':
            self.cnn = resnet50(output_channels=image_feature_dim)
        else:
            raise NotImplementedError()
        self.conv = nn.Conv2d(image_feature_dim, out_dim, self.roi_pooling_size)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(out_dim)

        position_embedding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, max_len, 2).float() * -(math.log(10000.0) / d_model))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', position_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor, boxes_coordinate: torch.Tensor,
                transcripts: torch.Tensor, src_key_padding_mask: torch.Tensor):
        """
        images: [B, C, H, W]
        boxes_coordinate: [B, N, 8]
        transcripts: [B, N, T, D]
        src_key_padding_mask:
        """
        b, n, t, d = transcripts.shape
        _, _, h_origin, w_origin = images.shape  # [B, C, H, W]
        images = self.cnn(images)
        _, c, h, w = images.shape  # [B, image_features_dim, H/16, W/16]
        rois_batch = torch.zeros(b, n, 5, device=images.device)
        for i in range(b):
            doc_boxes = boxes_coordinate[i]
            pos = torch.stack([doc_boxes[:, 0], doc_boxes[:, 1], doc_boxes[:, 4], doc_boxes[:, 5]], dim=1)
            rois_batch[i, :, 1:5] = pos
            rois_batch[i, :, 0] = i
        spatial_scale = float(h / h_origin)
        if self.roi_pooling_mode == 'roi_align':
            image_segments = roi_align(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        else:
            image_segments = roi_pool(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        # image segments: [B * N, C, roi_pooling_size, roi_pooling_size]
        image_segments = self.relu(self.bn(self.conv(image_segments)))  # (B*N, D, 1, 1)
        image_segments = image_segments.squeeze()
        transcripts_segments = transcripts + self.pe[:, :, :transcripts.size(2), :]
        transcripts_segments = self.dropout(transcripts_segments)
        transcripts_segments = transcripts_segments.reshape(b * n, t, d)
        transcripts_segments = self.transformer_encoder(transcripts_segments, src_key_padding_mask)

        image_segments = image_segments.expand_as(transcripts_segments)
        out = image_segments + transcripts_segments

        out = self.norm(out)
        out = self.dropout(out)
        return out
