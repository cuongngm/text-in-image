import math
import torch
import torch.nn as nn
import torch.nn.functional as f


class MultiAspectGCAttention(nn.Module):
    def __init__(self, inplanes, ratio=0.0625, headers=8, pooling_type='att', att_scale=False,
                 fusion_type='channel_concat'):
        super().__init__()
        assert pooling_type in ['avg', 'att']
        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        self.inplanes = inplanes
        self.ratio = ratio
        self.headers = headers
        self.att_scale = att_scale
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type

        self.planes = int(inplanes * ratio)
        self.single_header_inplanes = int(inplanes / headers)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=(1, 1))
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_concat_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1)),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1))
        )
        self.cat_conv = nn.Conv2d(2 * self.inplanes, self.inplanes, kernel_size=(1, 1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            x = x.view(batch * self.headers, self.single_header_inplanes, height, width)  # B*headers, C', H, W
            input_x = x
            input_x = input_x.view(batch * self.headers, self.single_header_inplanes, height*width)
            input_x = input_x.unsqueeze(1)  # [B*headers, 1, C', H*W]

            # branch context_mask
            # x : [B*header, C', H, W]
            context_mask = self.conv_mask(x)  # [B*headers, 1, H, W]
            context_mask = context_mask.view(batch * self.headers, 1, height * width)  # [B*headers, 1, H*W]
            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask /= math.sqrt(self.single_header_inplanes)
            context_mask = self.softmax(context_mask)  # [B*headers, 1, H*W]
            context_mask = context_mask.unsqueeze(-1)  # [B*headers, 1, H*W, 1]
            context = torch.matmul(input_x, context_mask)  # [B*headers, 1, C', 1]
            context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)  # [B, header*C', 1, 1]

        else:
            context = self.avg_pool(x)  # B, C, 1, 1
        return context

    def forward(self, x):
        # x: [B, C, H, W]
        context = self.spatial_pool(x)  # [B, C, 1, 1]
        out = x
        if self.fusion_type == 'channel_mul':
            channel_mul_term = torch.sigmoid(self.channel_concat_conv(context))  # [B, C, 1, 1]
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            channel_add_term = self.channel_concat_conv(context)
            out = out + channel_add_term
        else:
            channel_concat_term = self.channel_concat_conv(context)  # [B, C, 1, 1]
            b1, c1, _, _ = channel_concat_term.shape
            b, c, h, w = out.shape
            out = torch.cat([out, channel_concat_term.expand(-1, -1, h, w)], dim=1)  # [B, 2*C, H, W]
            out = self.cat_conv(out)  # [B, C, H, W]
            out = f.layer_norm(out, [self.inplanes, h, w])
            out = f.relu(out)
        return out

