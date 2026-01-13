"""
Module: Noise Factory
Description:
    该模块是整个项目的“噪声生成中枢”。它统一了散斑噪声（Speckle Noise）的生成算法，
    确保预处理阶段（Numpy）与模型训练阶段（Tensor）使用完全一致的物理模型。
    这保证了实验的可复现性与论文描述的严谨对齐。
"""

import torch
import numpy as np


class SpeckleNoiseFactory:
    """
    散斑噪声工厂类

    物理背景：
    超声图像中的散斑噪声通常被建模为“乘性噪声”。与常见的加性高斯噪声不同，
    散斑噪声的强度与信号本身的幅值成正比，这模拟了超声波相干干扰的成像特性。
    """

    @staticmethod
    def add_speckle_noise(img, sigma_sq):
        """
        根据论文公式实现乘性散斑噪声: Y = X + X * N

        参数说明:
            img: 输入图像，支持两种格式：
                 1. Numpy Array (H, W) - 用于离线数据预处理
                 2. PyTorch Tensor (C, H, W) - 用于在线训练数据增强
            sigma_sq: 噪声方差 (论文中定义的 σ²)，控制噪声的严重程度。
                      常见取值：0.001 (轻微), 0.02 (中等), 0.5 (严重)。

        返回:
            与输入类型一致的含噪图像。
        """

        # --- 场景 A：处理 Numpy 格式 (可用于 step2_prepare_dataset.py) ---
        if isinstance(img, np.ndarray):
            # 1. 归一化：将 0-255 的整数转换为 0-1 的浮点数进行数学运算
            image_f = img.astype(np.float32) / 255.0

            # 2. 生成噪声 N：均值为 0，方差为 sigma_sq 的高斯分布
            # 注意：np.random.normal 的第二个参数是标准差，即 sqrt(sigma_sq)
            noise = np.random.normal(0, np.sqrt(sigma_sq), image_f.shape)

            # 3. 乘性合成：Y = X + X * N
            noisy = image_f + image_f * noise

            # 4. 后处理：裁剪 [0, 1] 范围防止溢出，并恢复至 8-bit (0-255) 整数格式
            return np.uint8(np.clip(noisy, 0, 1) * 255)

        # --- 场景 B：处理 PyTorch Tensor 格式 (用于训练时的在线增强) ---
        elif torch.is_tensor(img):
            # 1. 噪声生成：torch.randn_like 在 GPU 上生成与原图尺寸一致的正态分布随机数
            # 乘以 sqrt(sigma_sq) 调整其方差
            noise = torch.randn_like(img) * np.sqrt(sigma_sq)

            # 2. 乘性合成：利用张量广播并行计算
            # 对应公式：含噪像素 = 原始像素 + 原始像素 * 噪声
            noisy = img + img * noise

            # 3. 约束范围：将值限制在 [0.0, 1.0] 之间，保持数据分布稳定
            return torch.clamp(noisy, 0, 1)

        # 如果类型不匹配，原样返回
        return img
