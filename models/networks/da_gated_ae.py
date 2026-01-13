"""
Module: Network Architecture - DA-Gated AE (Proposed Method)
Description:
    该模块实现了论文提出的 DA-Gated AE。其核心创新点包括：
    1. Dual-Attention Gated (DA-Gate) 模块：顺序执行通道与空间注意力，过滤跳跃连接中的噪声。
    2. 轻量化策略：采用深度可分离卷积 (DSC) 降低参数量。
    3. 增强型解码器：在 Expansive Path 中引入 BN 层以稳定受散斑干扰的梯度流。
"""

import torch
import torch.nn as nn


# ==========================================
# 1. 基础轻量化组件 - 深度可分离卷积 (DSC)
# ==========================================
class DSCBlock(nn.Module):
    """
    Depthwise Separable Convolution.
    将空间卷积与通道变换解耦，显著降低参数量和计算量 (FLOPs)。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DSCBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# ==========================================
# 2. DA-Gate 子模块 - 通道注意力 (CAM)
# ==========================================
class CAModule(nn.Module):
    """
    Channel Attention Module.
    通过 GAP 和 GMP 提取通道描述符，并利用 Shared MLP 建模通道间的非线性依赖。
    """

    def __init__(self, channels, reduction=16):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP 结构
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out  # Channel-wise multiplication


# ==========================================
# 3. DA-Gate 子模块 - 空间注意力 (SAM)
# ==========================================
class SAModule(nn.Module):
    """
    Spatial Attention Module.
    沿通道维度进行池化，通过 7x7 大核卷积捕捉空间结构特征。
    """

    def __init__(self, kernel_size=7):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.sigmoid(self.conv(res))
        return x * res  # Spatial-wise multiplication


# ==========================================
# 4. 核心组件 - DA-Gate Module
# ==========================================
class DAGate(nn.Module):
    """
    Dual-Domain Attention-Gated Module.
    采用 "Channel-first, Spatial-second" 的顺序校准逻辑。
    """

    def __init__(self, channels):
        super(DAGate, self).__init__()
        self.cam = CAModule(channels)
        self.sam = SAModule()

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x


# ==========================================
# 5. 增强型上采样组件
# ==========================================
class UpsampleBlock(nn.Module):
    """
    专门针对扩张路径设计的上采样块，集成 BN 层。
    """
    def __init__(self, in_ch, out_ch):
        super(UpsampleBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch), # ✅ 强化点：在上采样处也加入 BN
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class DAGatedAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_feat=32):
        super(DAGatedAE, self).__init__()

        # --- Step 1: Lightweight Encoding ---
        self.enc1 = self._make_dsc_layer(in_channels, base_feat)
        self.enc2 = self._make_dsc_layer(base_feat, base_feat * 2)
        self.enc3 = self._make_dsc_layer(base_feat * 2, base_feat * 4)
        self.enc4 = self._make_dsc_layer(base_feat * 4, base_feat * 8)  # Bottleneck

        self.pool = nn.MaxPool2d(2)

        # --- Step 2: DA-Gate ---
        self.gate1 = DAGate(base_feat)
        self.gate2 = DAGate(base_feat * 2)
        self.gate3 = DAGate(base_feat * 4)

        # --- Step 3: Decoding ---
        self.up3 = nn.UpsampleBlock(base_feat * 8, base_feat * 4)
        self.dec3 = self._make_dsc_layer(base_feat * 8, base_feat * 4)  # base*4(up) + base*4(gate)

        self.up2 = nn.UpsampleBlock(base_feat * 4, base_feat * 2)
        self.dec2 = self._make_dsc_layer(base_feat * 4, base_feat * 2)

        self.up1 = nn.UpsampleBlock(base_feat * 2, base_feat)
        self.dec1 = self._make_dsc_layer(base_feat * 2, base_feat)

        self.final_conv = nn.Conv2d(base_feat, out_channels, kernel_size=1)

    def _make_dsc_layer(self, in_ch, out_ch):
        """算法要求每个 level 包含 DSC 块"""
        # 按照论文通常实践，每个 Level 包含两层卷积以增加深度
        return nn.Sequential(
            DSCBlock(in_ch, out_ch),
            DSCBlock(out_ch, out_ch)
        )

    def forward(self, x):
        # Step 1: Encoding
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        b = self.enc4(p3)

        # Step 2 & 3: Gating & Decoding
        # Level 3
        g3 = self.gate3(e3)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, g3], 1))

        # Level 2
        g2 = self.gate2(e2)
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, g2], 1))

        # Level 1
        g1 = self.gate1(e1)
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, g1], 1))

        # Final Step
        return self.final_conv(d1)