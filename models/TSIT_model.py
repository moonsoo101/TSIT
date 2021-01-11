from typing import Any

import torch
import torch.nn as nn


class TSIT(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.netG = Generator()
        self.netD = Discriminator()
        self.netCS = CSStream().getStreamResBlks()
        self.netSS = CSStream().getStreamResBlks()

    def forward(self, x):
        return self.netCS(x)


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return super().forward(x)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return super().forward(x)


class CSStream:

    def __init__(self):
        super().__init__()
        self.feature_shape = [64, 128, 256, 512, 1024, 1024, 1024, 1024]
        self.n_blocks = 7
        self.input_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1)
        self.StreamResBlks = nn.ModuleList([StreamResBlk(self.feature_shape[i], self.feature_shape[i+1]) for i in range(self.n_blocks)])

    def getStreamResBlks(self):
        return self.StreamResBlks


class ResBlk(nn.Module):

    def __init__(self, inc, outc, ks, pad, first=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ks, stride=1, padding=pad)
        self.norm = nn.InstanceNorm2d(outc)
        self.LReLU = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.norm(x1)
        out = self.LReLU(x2)
        return out


class StreamResBlk(nn.Module):

    def __init__(self, inc, outc):
        super().__init__()
        self.resBlk1 = ResBlk(inc, inc, 3, 1)
        self.skip_connection = ResBlk(inc, outc, 1, 0)
        self.resBlk2 = ResBlk(inc, outc, 3, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='nearest')
        x1, sc = self.resBlk1(x), self.skip_connection(x)
        x2 = self.resBlk2(x1)
        out = x2 + sc
        return out