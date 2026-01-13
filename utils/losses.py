import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class HybridLoss(nn.Module):
    """
    【核心模块：DA-Gated AE 复合损失函数】

    设计动机：
    超声图像去噪不仅要去除噪声（重构），还要保持卵泡的解剖结构（结构相似性）
    以及清晰的边界（边缘一致性）。单一的 MSE 损失会导致结果过于模糊。

    计算公式:
    L_total = λ1 * L_char + λ2 * L_ssim + λ3 * L_edge
    """

    def __init__(self, lambda_rec=1.0, lambda_ssim=0.5, lambda_edge=0.1, epsilon=1e-3):
        """
        参数说明：
        :param lambda_rec:  重构损失权重 (λ1)。默认为 1.0，作为主导损失。
        :param lambda_ssim: 结构相似性权重 (λ2)。设定为 0.5，保护解剖形态。
        :param lambda_edge: 边缘损失权重 (λ3)。设定为 0.1，强化边缘清晰度。
        :param epsilon:     Charbonnier 惩罚项的微小常数，保证数值稳定性。
        """
        super(HybridLoss, self).__init__()
        self.l_rec = lambda_rec
        self.l_ssim = lambda_ssim
        self.l_edge = lambda_edge
        self.eps = epsilon

        # --- 预定义 Sobel 算子 (用于边缘检测) ---
        # 算子原理：通过计算像素点上下、左右的差值，提取图像的梯度
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 使用 register_buffer 将算子存入模型。
        self.register_buffer('kernel_x', kx)
        self.register_buffer('kernel_y', ky)

    def charbonnier_loss(self, x, y):
        """
        【分量 1：Charbonnier 重构损失】
        """
        # 公式：sqrt((x - y)^2 + eps^2)
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))

    def edge_consistency_loss(self, x, y):
        """
        【分量 2：边缘一致性损失】
        强制去噪后的图像与原始干净图像在“梯度图”上保持一致，
        从而避免超声图像中重要的卵泡壁被过度平滑掉。
        """
        # 计算预测图在 X 和 Y 方向的梯度
        gx_x = F.conv2d(x, self.kernel_x, padding=1)
        gy_x = F.conv2d(x, self.kernel_y, padding=1)

        # 计算真实图在 X 和 Y 方向的梯度
        gx_y = F.conv2d(y, self.kernel_x, padding=1)
        gy_y = F.conv2d(y, self.kernel_y, padding=1)

        # 惩罚两个梯度图之间的差异
        loss_edge = torch.mean(torch.abs(gx_x - gx_y)) + torch.mean(torch.abs(gy_x - gy_y))
        return loss_edge

    def forward(self, pred, target):
        """
        前向传播计算总损失
        :param pred:   模型预测的去噪图 (Batch, 1, 256, 256)
        :param target: 对应的干净图 Label (Batch, 1, 256, 256)
        """
        # 1. 计算鲁棒重构损失 (L_char)
        l_c = self.charbonnier_loss(pred, target)

        # 2. 计算结构相似性损失 (L_ssim)
        # ssim() 返回 [0, 1]，1 表示完全相同。用 1 减去它得到损失值。
        l_s = 1 - ssim(pred, target, data_range=1.0, size_average=True)

        # 3. 计算边缘一致性损失 (L_edge)
        l_e = self.edge_consistency_loss(pred, target)

        # 4. 加权求和得到最终 Loss
        total_loss = self.l_rec * l_c + self.l_ssim * l_s + self.l_edge * l_e

        return total_loss
