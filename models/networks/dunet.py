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


class DenseBlock(nn.Module):
    """
    密集连接块 (Dense Block)。
    """

    def __init__(self, in_channels, growth_rate):
        """
        Args:
            in_channels (int): 输入通道数。
            growth_rate (int): 每层卷积新生成的特征通道数。
        """
        super(DenseBlock, self).__init__()

        # 第一层卷积：提取基础特征
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)

        # 第二层卷积：输入是 (原始输入 + 第一层输出) 的拼接
        # 因此输入通道数为 in_channels + growth_rate
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播逻辑：
        x -> conv1 -> out1
        cat(x, out1) -> conv2 -> out2
        return cat(x, out1, out2)
        """
        # 第一路
        out1 = self.relu(self.conv1(x))

        # 第二路：在维度 1 (Channel) 上进行拼接
        # 假设 x 维度是 [B, 32, H, W], out1 是 [B, 32, H, W]
        # 拼接后的 input2 维度是 [B, 64, H, W]
        input2 = torch.cat([x, out1], 1)
        out2 = self.relu(self.conv2(input2))

        # 最终输出：[B, in + 2*growth_rate, H, W]
        return torch.cat([x, out1, out2], 1)


class DUNet(nn.Module):
    """
    DU-Net 核心架构。
    """

    def __init__(self, in_channels=1, out_channels=1, features=32):
        """
        Args:
            in_channels (int): 输入图像通道，超声灰度图通常为 1。
            out_channels (int): 输出图像通道，通常为 1。
            features (int): 初始特征图通道数（也决定了后续 DenseBlock 的增长率）。
        """
        super(DUNet, self).__init__()

        # --- 1. Encoder (编码器/提取段) ---
        # 作用：初步提取空间特征。输入 1 -> 输出 32 -> 经过 DenseBlock 变为 32 * 3 = 96
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DenseBlock(features, features)
        )

        # --- 2. Middle (中间段/瓶颈层) ---
        # 作用：通过下采样减少空间维度，增大感受野。
        # MaxPool2d 之后：H, W 减半。
        # DenseBlock 输入 96 -> 输出 96 + 32*2 = 160
        # ConvTranspose2d (转置卷积/反卷积)：将 160 还原回 96，并让 H, W 翻倍。
        self.middle = nn.Sequential(
            nn.MaxPool2d(2),
            DenseBlock(features * 3, features),
            nn.ConvTranspose2d(features * 5, features * 3, kernel_size=2, stride=2)
        )

        # --- 3. Decoder (解码器/重建段) ---
        # 作用：将提取出的深层特征映射回原始图像空间。
        # 输入 96 -> 经过 DenseBlock 变为 160 -> 最后 1x1 卷积降维至 1 (out_channels)
        self.decoder = nn.Sequential(
            DenseBlock(features * 3, features),
            nn.Conv2d(features * 5, out_channels, kernel_size=1)
        )

    def forward(self, x):
        """
        全流程前向传播。
        Tensor 维度跟踪:
        input: [B, 1, H, W]
        enc:   [B, 96, H, W]
        mid:   [B, 96, H, W] (经历过先缩小再放大)
        out:   [B, 1, H, W]
        """
        enc = self.encoder(x)
        mid = self.middle(enc)
        out = self.decoder(mid)
        return out