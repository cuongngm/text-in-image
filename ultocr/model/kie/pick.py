from typing import *
import numpy as np
import torch
import torch.nn as nn

from ultocr.model.kie.encoder import Encoder
from ultocr.model.kie.graph import GLCN


class PICK(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder()
        self.graph = GLCN(in_dim=512, out_dim=512)

    def aggregate_avg_pooling(self, input_tensor, text_mask):
        """
        input_tensor: [B* N, T, D]
        text_mask: [B*N, T]
        """
        input_tensor = input_tensor * text_mask.detach().unsqueeze(2).float()
        sum_out = torch.sum(input_tensor, dim=1)  # [B*N, D]
        text_len = text_mask.float().sum(dim=1)  # [B*N, ]
        text_len = text_len.unsqueeze(1).expand_as(sum_out)  # [B*N, D]
        text_len += text_len.eq(0).float()
        mean_out = sum_out.div(text_len)  # [B*N, D]
        return mean_out

    @staticmethod
    def compute_mask(mask: torch.Tensor):
        b, n, t = mask.shape
        mask = mask.reshape(b * n, t)
        mask_sum = mask.sum(dim=-1)  # [b*n,]
        graph_node_mask = mask_sum != 0
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(b * n, t)
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  # [b * n, t]
        return src_key_padding_mask, graph_node_mask

    def forward(self, whole_image: torch.Tensor, relation_features: torch.Tensor,
                text_segments: torch.Tensor, text_length: torch.Tensor,
                mask: torch.Tensor, boxes_coordinate: torch.Tensor, iob_tags_label: torch.Tensor):
        """
        whole_image: [B, 3, H, W]
        relation_features: [B, N, N, 6]
        text_segments: [B, N, T]
        text_length: [B, N]
        iob_tags_label: [B, N, T]
        mask: [B, N, T]
        boxes_coordinate: [B, N, 8]
        """
        # Encoder module
        # word embedding
        text_emb = self.word_emb(text_segments)
        # src_key_padding_mask is text padding mask (B*N, T)
        # graph_node_mask is mask for graph, (B*N, T)
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)
        x = self.encoder(whole_image, boxes_coordinate, text_segments, src_key_padding_mask)  # [B*N, T, D]

        # graph module
        text_mask = torch.logical_not(src_key_padding_mask).byte()
        x_gcn = self.aggregate_avg_pooling(x, text_mask)  # [B*N, D]
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)  # [B*N, 1]
        x_gcn = x_gcn * graph_node_mask.byte()  # [B*N, D]

        # initial adjacent matrix (b, n, n)
        b, n, t = mask.shape
        init_adj = torch.ones((b, n, n), device=text_emb.device)
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)  # [B, 1]
        x_gcn = x_gcn.reshape(b, n, -1)  # [B, N, D]
        x_gcn, soft_adj, gl_loss = self.graph(x_gcn, relation_features, init_adj, boxes_num)
        adj = soft_adj * init_adj

        # decode module
