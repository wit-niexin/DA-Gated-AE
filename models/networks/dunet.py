"""
Module: Network Architecture - DU-Net (Dense U-Net variant)
Description:
    该模块定义了去噪网络的核心结构。
    采用了密集连接块 (DenseBlock) 替代传统的卷积层，以增强特征传播并减少梯度消失。
    该架构通过 Encoder 提取多尺度特征，Middle 层进行下采样与上采样变换，最后由 Decoder 重建灰度图像。

Key Components:
    - DenseBlock: 实现密集连接，通过通道拼接 (Concatenation) 实现特征重用。
    - DUNet: 整体采用对称式结构，输入为 [B, 1, H, W] 的含噪图，输出为相同尺寸的去噪图。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        input2 = torch.cat([x, out1], 1)
        out2 = self.conv2(input2)
        return torch.cat([x, out1, out2], 1)


class DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(DUNet, self).__init__()

        # Encoder
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.db1 = DenseBlock(features, features)  # 32 -> 96
        self.pool1 = nn.MaxPool2d(2)

        self.db2 = DenseBlock(96, features)  # 96 -> 160
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.db_bot = DenseBlock(160, features)  # 160 -> 224

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce2 = nn.Conv2d(224 + 160, 160, 1)  # 拼接 e2
        self.db_dec2 = DenseBlock(160, features)  # 160 -> 224

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce1 = nn.Conv2d(224 + 96, 96, 1)  # 拼接 e1
        self.db_dec1 = DenseBlock(96, features)  # 96 -> 160

        self.tail = nn.Conv2d(160, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.head(x)
        e1 = self.db1(x1)  # [B, 96, H, W]

        p1 = self.pool1(e1)
        e2 = self.db2(p1)  # [B, 160, H/2, W/2]

        p2 = self.pool2(e2)

        # Bottleneck
        b = self.db_bot(p2)  # [B, 224, H/4, W/4]

        # Decoder Level 2
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)  # 第一次拼接
        d2 = self.reduce2(d2)
        d2 = self.db_dec2(d2)  # [B, 224, H/2, W/2]

        # Decoder Level 1
        d1 = self.up1(d2)
        # 尺寸对齐
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)

        d1 = torch.cat([d1, e1], dim=1)  # 第二次拼接 (论文批判的噪声泄露点)
        d1 = self.reduce1(d1)
        d1 = self.db_dec1(d1)  # [B, 160, H, W]

        return self.tail(d1)
