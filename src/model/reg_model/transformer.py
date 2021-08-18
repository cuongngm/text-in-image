import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as f


def clones(to_clone_module, clone_time, is_deep=True):
    copy_method = copy.deepcopy if is_deep else copy.copy
    return nn.ModuleList([copy_method(to_clone_module) for _ in range(clone_time)])


class MultiHeadAttention(torch.jit.ScriptModule):
    def __init__(self, nhead, d_model, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.h = nhead
        self.d_k = int(d_model / nhead)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = None
        self.dropout = nn.Dropout(dropout)

    @torch.jit.script_method
    def attention(self, q, k, v, mask):
        """
        q, k, v : [N, h, seq_len, d_model]
        """
        d_k = v.size(-1)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [N, h, seq_len, seq_len]
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # [N, h, seq_len, seq_len]
        attn = f.softmax(score, dim=-1)
        return torch.matmul(attn, v), attn

    @torch.jit.script_method
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (q, k, v))]
        product_and_attention = self.attention(q, k, v, mask=mask)
        x = product_and_attention[0]  # N, seq_len, d_model
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h*self.d_k)
        return self.linears[-1](x)


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, ff_dim)
        self.w_2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class PositionalEncoding(torch.jit.ScriptModule):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    @torch.jit.script_method
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, with_encoder, nhead, d_model, num_layer, dropout, dim_ff, share_parameter=True):
        super().__init__()
        self.with_encoder = with_encoder
        self.num_layer = num_layer
        self.share_parameter = share_parameter
        self.attention = nn.ModuleList([
            MultiHeadAttention(nhead, d_model, dropout)
            for _ in range(1 if self.share_parameter else num_layer)
        ])
        self.feed_forward = nn.ModuleList([
            FeedForward(d_model, dim_ff, dropout)
            for _ in range(1 if self.share_parameter else num_layer)
        ])

        self.position = PositionalEncoding(d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, x):
        tgt_length = x.size(1)
        return torch.ones((tgt_length, tgt_length))

    def forward(self, input_tensor):
        output = self.position(input_tensor)
        if self.with_encoder:
            src_mask = self.generate_mask(output)
            for i in range(self.num_layer):
                actual_i = 0 if self.share_parameter else i
                norm_output = self.layer_norm(output)
                output += self.dropout(self.attention[actual_i](norm_output, norm_output, norm_output, src_mask))
                norm_output = self.layer_norm(output)
                output += self.dropout(self.feed_forward[actual_i](norm_output))
            output = self.layer_norm(output)
        return output


class Decoder(nn.Module):
    def __init__(self, nhead, d_model, num_layer, dropout, ff_dim, n_class, padding_symbol=0, share_parameter=False):
        super().__init__()
        self.share_parameter = share_parameter
        self.attention = nn.ModuleList([
            MultiHeadAttention(nhead, d_model, dropout)
            for _ in range(1 if share_parameter else num_layer)
        ])
        self.src_attention = nn.ModuleList([
            MultiHeadAttention(nhead, d_model, dropout)
            for _ in range(1 if share_parameter else num_layer)
        ])
        self.feed_forward = nn.ModuleList([
            FeedForward(d_model, ff_dim, dropout)
            for _ in range(1 if share_parameter else num_layer)
        ])
        self.position = PositionalEncoding(d_model, dropout)
        self.num_layer = num_layer
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.embedding = nn.Embedding(n_class, d_model)
        self.padding_symbol = padding_symbol
        self.sqrt_model_size = math.sqrt(d_model)

    def generate_tgt_mask(self, src, tgt):
        tgt_pad_mask = (tgt != self.padding_symbol).unsqueeze(1).unsqueeze(3)  # [b, 1, len_src, 1]
        tgt_length = tgt.size(1)
        tgt_sub_mask = torch.tril(
            torch.ones((tgt_length, tgt_length), dtype=torch.uint8)
        )
        src_mask = torch.ones((tgt_length, src.size(1)), dtype=torch.uint8)
        tgt_mask = tgt_pad_mask & tgt_sub_mask.bool()
        return src_mask, tgt_mask

    def forward(self, tgt_result, memory):
        tgt = self.embedding(tgt_result) * self.sqrt_model_size
        tgt = self.position(tgt)
        src_mask, tgt_mask = self.generate_tgt_mask(memory, tgt_result)
        output = tgt
        for i in range(self.num_layer):
            actual_i = 0 if self.share_parameter else i
            norm_output = self.layer_norm(output)
            output += self.dropout(self.attention[actual_i](norm_output, norm_output, norm_output, tgt_mask))
            norm_output = self.layer_norm(output)
            output += self.dropout(self.src_attention[actual_i](norm_output, memory, memory, src_mask))
            norm_output = self.layer_norm(output)
            output += self.dropout(self.feed_forward[actual_i](norm_output))
        return self.layer_norm(output)
