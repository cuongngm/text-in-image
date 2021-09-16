import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from ultocr.model.common.context_block import MultiAspectGCAttention


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_gcb=False):
        super().__init__()
        # kernel size: 3, 3
        # stride: stride, 1
        # padding: 1, 1
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.use_gcb = use_gcb
        if self.use_gcb:
            self.context_block = MultiAspectGCAttention(inplanes=out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_gcb:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_gcb=False):
        super().__init__()
        # kernel size: 1, 3, 1
        # stride: 1, stride, 1
        # padding: 0, 1, 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.downsample = downsample
        self.use_gcb = use_gcb
        if self.use_gcb:
            self.context_block = MultiAspectGCAttention(inplanes=out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_gcb:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Resnet(nn.Module):
    # resnet from https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/faster_rcnn/resnet.py
    def __init__(self, block, layers, out_channels=512):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.conv2 = nn.Conv2d(512 * block.expansion, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ResnetMaster(nn.Module):
    # Resnet with global context block
    def __init__(self, block, layers, zero_init_residual=False, gcb_config=None):
        super().__init__()
        self.in_channels = 128
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self.make_layer(block, 256, layers[0], stride=1,
                                      use_gcb=gcb_config['model_arch']['common']['use_gcb'][0])
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = self.make_layer(block, 256, layers[1], stride=1,
                                      use_gcb=gcb_config['model_arch']['common']['use_gcb'][1])
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.layer3 = self.make_layer(block, 512, layers[2], stride=1,
                                      use_gcb=gcb_config['model_arch']['common']['use_gcb'][2])
        self.conv5 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)

        self.layer4 = self.make_layer(block, 512, layers[3], stride=1,
                                      use_gcb=gcb_config['model_arch']['common']['use_gcb'][3])
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def make_layer(self, block, out_channels, blocks, stride=1, use_gcb=False):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample,
                            use_gcb=use_gcb))
        self.in_channels = out_channels*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.max_pool3(x)

        x = self.layer2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.max_pool4(x)

        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        return x


def resnet18(output_channels=512):
    model = Resnet(BasicBlock, layers=[2, 2, 2, 2], out_channels=output_channels)
    return model


def resnet50(output_channels=512):
    model = Resnet(BottleNeck, layers=[3, 4, 6, 3], out_channels=output_channels)
    return model


def resnet50_master(gcb_config):
    model = ResnetMaster(BasicBlock, layers=[1, 2, 5, 3], gcb_config=gcb_config)
    return model


class ConvEmbeddingGC(nn.Module):
    def __init__(self, gcb_config):
        super().__init__()
        self.backbone = resnet50_master(gcb_config)

    def forward(self, x):
        feature = self.backbone(x)
        # print('feature', feature.size())
        b, c, h, w = feature.shape
        feature = feature.view(b, c, h*w)
        feature = feature.permute((0, 2, 1))
        return feature
