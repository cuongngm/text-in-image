from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as f


class GraphLearningLayer(nn.Module):
    def __init__(self, in_dim: int, learning_dim: int, gamma: float, eta: float):
        super().__init__()
        self.learn_w = nn.Parameter(torch.empty(learning_dim))
        self.projection = nn.Linear(in_dim, learning_dim)
        self.gamma = gamma
        self.eta = eta

    def init_parameters(self):
        nn.init.uniform_(self.learn_w, a=0, b=1)

    @staticmethod
    def compute_dynamic_mask(box_num: torch.Tensor):
        """
        compute mask, if node(box) is not exist, the length of mask is calculate by max(box_num)
        will help with multi nodes multi gpu training, ensure batch of different gpu is same shape
        :param box_num: (B, 1)
        :return (B, N, N, 1)
        """
        max_len = torch.max(box_num)
        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))  # [B, N]
        box_num = box_num.expand_as(mask)
        mask = mask < box_num  # [B, N]

        row_mask = mask.unsqueeze(1)  # [B, 1, N]
        column_mask = mask.unsqueeze(2)  # [B, N, 1]
        mask = (row_mask & column_mask)  # [B, N, N]
        mask = ~mask * -1  # -1 if not exist node, or 0
        return mask.unsqueeze(-1)  # [B, N, N, 1]

    def graph_learning_loss(self, x_hat: torch.Tensor, adj: torch.Tensor, box_num: torch.Tensor):
        """
        calculate graph learning loss
        :param x_hat: (B, N, D)
        :param adj: (B, N, N)
        :param box_num: (B, 1)
        :return:
        gl_loss
        """
        b, n, d = x_hat.shape
        x_i = x_hat.unsqueeze(2).expand(b, n, n, d)
        x_j = x_hat.unsqueeze(1).expand(b, n, n, d)
        box_num_div = 1. / torch.pow(box_num.float(), 2)  # [B, 1]
        dist_loss = adj + self.eta * torch.norm(x_i - x_j, dim=3)
        dist_loss = torch.exp(dist_loss)
        dist_loss = torch.sum(dist_loss, dim=(1, 2)) * box_num_div.squeeze(-1)
        f_norm = torch.norm(adj, dim=(1, 2))
        gl_loss = dist_loss + self.gamma * f_norm
        return gl_loss

    def forward(self, x: torch.Tensor, adj: torch.Tensor, box_num: torch.Tensor = None):
        """
        x: nodes set (B, N, D)
        adj: init adj (B, N, N, in_dim)
        box_num: (B, 1)
        return: out, soft adj matrix, gl loss
        """
        b, n, d = x.shape
        x_hat = self.projection(x)
        _, _, learning_dim = x_hat.shape
        x_i = x_hat.unsqueeze(2).expand(b, n, n, learning_dim)
        x_j = x_hat.unsqueeze(1).expand(b, n, n, learning_dim)
        distance = torch.abs(x_i - x_j)
        # add -1 flag to distance if node is not exist. To seperate normal node distances from not exist node distance.
        if box_num is not None:
            mask = self.compute_dynamic_mask(box_num)
            distance = distance + mask
        distance = torch.einsum('bijd, d->bij', distance, self.learn_w)  # [B, N, N]
        out = f.leaky_relu(distance)
        # for numerical stability, due to softmax operation mable produce large value
        max_out_v, _ = out.max(dim=-1, keepdim=True)
        out = out - max_out_v
        # compute soft adjacent matrix A
        soft_adj = torch.exp(out)
        soft_adj = adj * soft_adj
        sum_out = soft_adj.sum(dim=-1, keepdim=True)
        soft_adj = soft_adj / sum_out + 1e-10
        gl_loss = None
        if self.training:
            gl_loss = self.graph_learning_loss(x_hat, soft_adj, box_num)
        return soft_adj, gl_loss


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_alpha = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w_vi = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w_vj = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias_h = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w_node = nn.Parameter(torch.empty(in_dim, out_dim))

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.w_alpha, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vi, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vj, a=math.sqrt(5))
        nn.init.uniform_(self.bias_h, a=0, b=1)
        nn.init.kaiming_uniform_(self.w_alpha, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, adj: torch.Tensor):
        """
        :param x: node embedding (B, N, in_dim)
        :param alpha: relation embedding (B, N, N, in_dim)
        :param adj: learned soft adjacent matrix (B, N, N)
        :param box_num: (B, 1)
        :return:
            x_out: updated node embedding (B, N, out_dim)
            alpha: updated relation embedding (B, N, N, out_dim)
        """
        b, n, in_dim = x.shape
        x_i = x.unsqueeze(2).expand(b, n, n, in_dim)
        x_j = x.unsqueeze(1).expand(b, n, n, in_dim)

        x_i = torch.einsum('bijd, dk->bijk', x_i, self.w_vi)
        x_j = torch.einsum('bijd, dk->bijk', x_j, self.w_vj)

        # update hidden features between nodes (B, N, N, in_dim)
        H = f.relu(x_i + x_j + alpha + self.bias_h)

        # update nodes embedding x (B, N, out_dim)
        AH = torch.einsum('bij, bijd->bid', adj, H)
        new_x = torch.einsum('bid, dk->bik', AH, self.w_node)
        new_x = f.relu(new_x)

        # update relation embedding [B, N, N, out_dim]
        new_alpha = torch.einsum('bijd, dk->bijk', H, self.w_alpha)
        new_alpha = f.relu(new_alpha)
        return new_x, new_alpha


class GLCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, gamma: float = 0.0001, eta: float = 1,
                 learning_dim: int = 128, num_layers=2):
        super().__init__()
        self.alpha_transform = nn.Linear(6, in_dim)
        self.gl_layer = GraphLearningLayer(in_dim, learning_dim, gamma, eta)
        in_dim_cur = in_dim
        modules = []
        for i in range(num_layers):
            m = GCNLayer(in_dim_cur, out_dim)
            in_dim_cur = out_dim
            out_dim = in_dim_cur
            modules.append(m)
        self.gcn = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, rel_features: torch.Tensor,
                adj: torch.Tensor, box_num: torch.Tensor):
        """
        x: nodes embedding: (B, N, D)
        rel_features: relation embedding (B, N, N, 6)
        adj: default adjacent matrix (B, N, N)
        box_num: (B, 1)
        """
        alpha = self.alpha_transform(rel_features)
        # compute soft adj and graph learning loss
        soft_adj, gl_loss = self.gl_layer(x, rel_features, box_num)
        adj = adj * soft_adj
        # stack gcn layer
        for i, gcn_layer in enumerate(self.gcn):
            x, alpha = gcn_layer(x, alpha, adj)
        return x, soft_adj, gl_loss
