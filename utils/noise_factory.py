"""
Module: Noise Factory (采用 Correlated Noise 定义)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class SpeckleNoiseFactory:
    """
    散斑噪声工厂类 - 增强版（包含空间相关性卷积）
    """

    @staticmethod
    def add_speckle_noise(img, sigma_sq):
        """
        合成模型：
        1. Noise Correlation: N_corr = N_raw * h (3x3 Mean Filter)
        2. Image Synthesis: X = Y + Y * N_corr
        """

        # --- 场景 A：处理 Numpy 格式 (离线预处理) ---
        if isinstance(img, np.ndarray):
            image_f: np.ndarray = img.astype(np.float32) / 255.0

            # Step 1: 生成原始噪声 N_raw
            # 均值为 0, 方差为 sigma_sq
            n_raw = np.random.normal(0, np.sqrt(sigma_sq), image_f.shape).astype(np.float32)

            # Step 2: 卷积产生相关性 (3x3 Mean Filtering Kernel)
            # cv2.blur 实现了与均值核卷积的等效操作
            n_corr = cv2.blur(n_raw, (3, 3))

            # Step 3: 乘性合成 X = Y + Y * N_corr
            noisy = image_f + image_f * n_corr

            return np.uint8(np.clip(noisy, 0, 1) * 255)

        # --- 场景 B：处理 PyTorch Tensor 格式 (在线训练) ---
        elif torch.is_tensor(img):
            # Step 1: 生成原始噪声 N_raw
            n_raw = torch.randn_like(img) * np.sqrt(sigma_sq)

            # Step 2: 卷积产生相关性 (使用 3x3 均值池化模拟卷积核 h)
            # padding=1 保证输出尺寸不变，stride=1 保证逐像素平滑
            if img.dim() == 3:  # (C, H, W)
                n_raw_4d = n_raw.unsqueeze(0)
                n_corr_4d = F.avg_pool2d(n_raw_4d, kernel_size=3, stride=1, padding=1)
                n_corr = n_corr_4d.squeeze(0)
            else:  # (B, C, H, W)
                n_corr = F.avg_pool2d(n_raw, kernel_size=3, stride=1, padding=1)

            # Step 3: 乘性合成
            noisy = img + img * n_corr

            return torch.clamp(noisy, 0, 1)

        return img
