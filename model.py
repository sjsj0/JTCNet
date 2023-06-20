# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import unetPart
sys.path.append('..')

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = unetPart.DoubleConv(n_channels, 48)
        self.down1 = unetPart.Down(48, 96)
        self.down2 = unetPart.Down(96, 192)
        self.down3 = unetPart.Down(192, 384)
        self.down4 = unetPart.Down(384, 384)
        self.up1 = unetPart.Up(768, 192, bilinear)
        self.up2 = unetPart.Up(384, 96, bilinear)
        self.up3 = unetPart.Up(192, 48, bilinear)
        self.up4 = unetPart.Up(96, 48, bilinear)
        self.outc = unetPart.OutConv(48, n_classes)
        self.HR = nn.Sequential(BasicBlock(384, 256, 2, downsample=1, Res=1),
                                BasicBlock(256, 256, 1, downsample=0, Res=0),
                                BasicBlock(256, 512, 2, downsample=1, Res=1),
                                BasicBlock(512, 512, 1, downsample=0, Res=0)
                                )
        self.av = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        HR = self.fc(torch.squeeze(self.av(self.HR(x4))))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        signals = self.outc(x)
        BVP_features = torch.squeeze(self.av(self.HR(x4)))
        return signals, torch.squeeze(HR),BVP_features


class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
                 )
        self.downsample = downsample
        self.Res = Res

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        return out


class rPPGNet(nn.Module):
    def __init__(self):
        super(rPPGNet, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.down = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlock(64, 64, 2, downsample=1),
            BasicBlock(64, 64, 1, downsample=1),
            BasicBlock(64, 128, 2, downsample=1),
            BasicBlock(128, 128, 1, downsample=1),
            BasicBlock(128, 256, 2, downsample=1, Res=1),
            BasicBlock(256, 256, 1, downsample=1),
            BasicBlock(256, 512, 2, downsample=1, Res=1),
            BasicBlock(512, 512, 1, downsample=1),
        )
        self.HR = nn.Sequential(BasicBlock(512, 256, 2, downsample=1, Res=1),
                                BasicBlock(256, 256, 1, downsample=0, Res=0),
                                BasicBlock(256, 512, 2, downsample=1, Res=1),
                                BasicBlock(512, 512, 1, downsample=0, Res=0)
                                )
        self.av = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 128, [2, 1], downsample=1),
            nn.ConvTranspose2d(128, 128, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(128, 64, [2, 1], downsample=1),
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 16, [2, 1], downsample=1),
            nn.ConvTranspose2d(16, 16, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(16, 1, [1, 1], downsample=1),
        )

    def forward(self, x):
        f = self.down(self.bn(x))
        rPPG = self.up(f).squeeze(dim=1)
        HR = self.fc(torch.squeeze(self.av(self.HR(f))))
        return rPPG, torch.squeeze(HR)

