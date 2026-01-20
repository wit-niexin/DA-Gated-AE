"""
Module: Network Architecture - US-DRUNet (Nakagami-based Residual Variant)
Description:
    该模块实现了论文中作为对比的 US-DRUNet。其核心逻辑在于：
    1. 统计先验嵌入：将 Nakagami 分布参数 (mu, omega) 作为额外的通道输入到每一个残差块。
    2. 残差学习：采用深层残差块堆叠，旨在通过残差映射学习噪声分布。
    3. 重型架构：相比于轻量化的 DA-Gated AE，本项目实现的 US-DRUNet 通道数更多，参数量显著增加。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NakagamiResidualBlock(nn.Module):
    """
    带 Nakagami 参数嵌入的残差块 (Nakagami-embedded Residual Block)。
    论文要求：将噪声水平图（统计参数）嵌入到每一层。
    """

    def __init__(self, in_channels, nak_channels=2):
        """
        Args:
            in_channels (int): 输入特征图通道数。
            nak_channels (int): Nakagami 参数通道数，通常为 2 (mu, omega)。
        """
        super(NakagamiResidualBlock, self).__init__()

        # 嵌入层：将 Nakagami 参数图与特征图融合
        # 卷积输入通道 = 特征通道 + 参数通道
        self.conv1 = nn.Conv2d(in_channels + nak_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, nak_map):
        """
        x: 特征图 [B, C, H, W]
        nak_map: 对应的 Nakagami 参数图 [B, 2, H, W]
        """
        residual = x

        # 1. 空间对齐 Nakagami Map (确保参数图尺寸与当前特征图一致)
        if nak_map.shape[2:] != x.shape[2:]:
            current_nak = F.interpolate(nak_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            current_nak = nak_map

        # 2. 拼接嵌入与残差运算
        out = torch.cat([x, current_nak], dim=1)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        # 3. 残差连接：F(x) + x
        out += residual
        return self.relu(out)


class USDRUNet(nn.Module):
    """
    US-DRUNet (Nakagami 变体) 完整架构。
    遵循深层堆叠的深层残差网络 (Deep Residual Network)。
    """

    def __init__(self, in_channels=1, out_channels=1, nak_channels=2, base_features=64):
        """
        Args:
            in_channels (int): 输入图通道 (1)。
            out_channels (int): 输出图通道 (1)。
            nak_channels (int): Nakagami (mu, omega) 通道 (2)。
            base_features (int): 基础特征数，US-DRUNet 通常设为 64 以模拟“重型”架构。
        """
        super(USDRUNet, self).__init__()

        # --- 1. 特征提取层 ---
        self.head = nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1)

        # --- 2. 编码器 (Encoder) ---
        # 相比 DU-Net，这里采用了更多的残差块堆叠来模拟论文中的“Heavy Stacking”
        self.layer1 = nn.Sequential(NakagamiResidualBlock(base_features), NakagamiResidualBlock(base_features))
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = nn.Sequential(NakagamiResidualBlock(base_features), NakagamiResidualBlock(base_features))
        self.pool2 = nn.MaxPool2d(2)

        # --- 3. 瓶颈层 (Bottleneck) ---
        self.bottleneck = nn.Sequential(NakagamiResidualBlock(base_features), NakagamiResidualBlock(base_features))

        # --- 4. 解码器 (Decoder) ---
        self.up2 = nn.ConvTranspose2d(base_features, base_features, kernel_size=2, stride=2)
        self.reduce2 = nn.Conv2d(base_features * 2, base_features, 1)
        self.layer3 = NakagamiResidualBlock(base_features, nak_channels)

        self.up1 = nn.ConvTranspose2d(base_features, base_features, kernel_size=2, stride=2)
        self.reduce1 = nn.Conv2d(base_features * 2, base_features, 1)
        self.layer4 = NakagamiResidualBlock(base_features, nak_channels)

        # --- 5. 重建层 ---
        self.tail = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x, nak_map):
        """
        Args:
            x: 原始输入图像 [B, 1, H, W]
            nak_map: Nakagami 统计图 [B, 2, H, W]，包含 mu 和 omega 信息。
        """
        feat = self.head(x)

        # Encoder
        e1 = feat
        for block in self.layer1: 
            e1 = block(e1, nak_map)
        
        p1 = self.pool1(e1)
        
        e2 = p1
        for block in self.layer2: 
            e2 = block(e2, nak_map)
        
        p2 = self.pool2(e2)

        # Bottleneck
        b = p2
        for block in self.bottleneck: 
            b = block(b, nak_map)

        # Decoder
        u2 = self.up2(b)
 
        # 尺寸对齐 (防止奇数尺寸报错)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=False)
            
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.reduce2(cat2)
        d2 = self.layer3(d2, nak_map)

        u1 = self.up1(d2)
        
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)

        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.reduce1(cat1)
        d1 = self.layer4(d1, nak_map)

        return self.tail(d1)


# ==========================================
# 工具函数：统计参数量
# ==========================================
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}