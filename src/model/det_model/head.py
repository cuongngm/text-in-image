import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, with_relu=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.with_relu = with_relu

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.with_relu:
            out = self.relu(out)
        return out


def upsample(x, y, scale=1):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H // scale, W // scale), mode='nearest')


def upsample_add(x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='nearest') + y


class DBHead(nn.Module):
    def __init__(self, backbone_out_channels, inner_channels=256, bias=False):
        # backbone_out_channels [256, 512, 1024, 2048]
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        # inner_channels = inner_channels // 4
        # reduce layers
        self.in5 = ConvBnRelu(backbone_out_channels[-1], inner_channels, 1, 1, 0, bias=bias)
        self.in4 = ConvBnRelu(backbone_out_channels[-2], inner_channels, 1, 1, 0, bias=bias)
        self.in3 = ConvBnRelu(backbone_out_channels[-3], inner_channels, 1, 1, 0, bias=bias)
        self.in2 = ConvBnRelu(backbone_out_channels[-4], inner_channels, 1, 1, 0, bias=bias)
        # smooth layers
        self.out5 = ConvBnRelu(inner_channels, inner_channels//4, 3, 1, 1, bias=bias)
        self.out4 = ConvBnRelu(inner_channels, inner_channels//4, 3, 1, 1, bias=bias)
        self.out3 = ConvBnRelu(inner_channels, inner_channels//4, 3, 1, 1, bias=bias)
        self.out2 = ConvBnRelu(inner_channels, inner_channels//4, 3, 1, 1, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.upsample_add(in5, in4)  # 1/16
        out3 = self.upsample_add(out4, in3)  # 1/8
        out2 = self.upsample_add(out3, in2)  # 1/4

        p5 = self.upsample(self.out5(in5), out2)
        p4 = self.upsample(self.out4(out4), out2)
        p3 = self.upsample(self.out3(out3), out2)
        p2 = self.out2(out2)
        fuse = torch.cat((p5, p4, p3, p2), 1)
        return fuse

    @staticmethod
    def upsample_add(x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    @staticmethod
    def upsample(x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='nearest')