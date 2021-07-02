from collections import OrderedDict
import torch
import torch.nn as nn


class SegDetector(nn.Module):
    def __init__(self,
                 inner_channels=256, k=10, adaptive=False,
                 serial=False, bias=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid()
            )
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial,bias=bias)
            self.thresh.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1),
            nn.Sigmoid()
            )
        return self.thresh

    def _init_upsample(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, fuse, img):
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                    (fuse, nn.functional.interpolate(
                        binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class SegDetectorVer1(nn.Module):
    def __init__(self, in_channels, k=50, adaptive=False, serial=False, bias=False, *args, **kwargs):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=bias),
                                      nn.BatchNorm2d(in_channels // 4),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
                                      nn.BatchNorm2d(in_channels // 4),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
                                      nn.Sigmoid())
        self.binarize.apply(self.weights_init)
        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(in_channels, serial=serial, bias=bias)
            self.thresh.apply(self.weights_init)

    def forward(self, fuse):
        # prob map / threshold map / appro binary map
        binary = self.binarize(fuse)
        thresh = self.thresh(fuse)
        if self.training and self.adaptive:
            # thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            y = torch.cat((binary, thresh, thresh_binary), dim=1)
        else:
            y = torch.cat((binary, thresh), dim=1)
        return y

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
                                    nn.BatchNorm2d(inner_channels // 4),
                                    nn.ReLU(inplace=True),
                                    self._init_upsample(inner_channels // 4, inner_channels // 4),
                                    nn.BatchNorm2d(inner_channels // 4),
                                    nn.ReLU(inplace=True),
                                    self._init_upsample(inner_channels // 4, 1),
                                    nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))