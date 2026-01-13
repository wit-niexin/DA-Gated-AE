"""
Module: Metrics
Description:
    图像去噪性能评价指标模块。包含全参考指标（PSNR, SSIM）和无参考指标（ENL, CNR）。
    全参考指标用于衡量去噪图像与 Ground Truth 之间的接近程度；
    无参考指标（ENL, CNR）则专门针对超声散斑噪声，评估模型对纹理保持和对比度增强的效果。

Key Metrics:
    - PSNR: Peak Signal-to-Noise Ratio (峰值信噪比)，数值越高代表失真越小。
    - SSIM: Structural Similarity (结构相似性)，数值越接近 1 代表结构保持越好。
    - ENL: Equivalent Number of Looks (等效视数)，评估平滑区域的去噪强度。
    - CNR: Contrast-to-Noise Ratio (对比度信噪比)，评估病灶/目标与背景的区分度。
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func


def calculate_psnr(img_true, img_test, data_range=255):
    """
    计算峰值信噪比 (PSNR)。
    基于均方误差 (MSE) 定义，是衡量重构质量最常用的指标。

    Args:
        img_true (ndarray): 参考图 (Clean/Ground Truth)。
        img_test (ndarray): 待评价图 (Denoised)。
        data_range (int): 像素值范围，默认为 255 (8-bit)。
    Returns:
        float: PSNR 分贝值 (dB)。
    """
    return psnr_func(img_true, img_test, data_range=data_range)


def calculate_ssim(img_true, img_test, data_range=255):
    """
    计算结构相似性 (SSIM)。
    通过亮度、对比度和结构三个维度的组合，比 PSNR 更符合人类视觉感知。

    Args:
        img_true (ndarray): 参考图。
        img_test (ndarray): 待评价图。
    Returns:
        float: SSIM 值，范围 [0, 1]。
    """
    return ssim_func(img_true, img_test, data_range=data_range)


def calculate_enl(image):
    """
    计算等效视数 (Equivalent Number of Looks, ENL)。

    超声图像特有指标。在同质区域（Uniform region），均值与方差的比值反映了散斑噪声的平滑程度。
    ENL 越大，说明算法对散斑噪声的抑制越彻底。

    Args:
        image (ndarray): 去噪后的图像。
    Returns:
        float: ENL 指标。
    """
    mu = np.mean(image)
    sigma = np.std(image)
    # 计算公式: (均值^2) / (方差^2)
    return (mu ** 2) / (sigma ** 2) if sigma != 0 else 0


def calculate_cnr(image):
    """
    计算对比度信噪比 (Contrast-to-Noise Ratio, CNR)。

    衡量图像中目标区域（如 PCOS 卵泡）与周围背景组织的对比清晰度。
    本实现采用 Otsu 自动阈值法将图像分为亮区和暗区，分别计算均值与标准差。

    Args:
        image (ndarray): 去噪后的图像。
    Returns:
        float: CNR 值。
    """
    # 1. 预处理：转换为 8 位图供 OpenCV 大津法处理
    img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. 自动阈值分割：将图像分为背景(Background)和目标(Target)
    # THRESH_OTSU 适合处理双峰直方图的超声图像
    ret, mask = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    target = image[mask == 0]  # 被分割出的暗色区域（通常是囊肿/卵泡内部）
    background = image[mask == 255]  # 被分割出的亮色区域（通常是基质组织）

    if len(target) == 0 or len(background) == 0:
        return 0

    # 3. 计算统计量
    mu_t, sigma_t = np.mean(target), np.std(target)
    mu_b, sigma_b = np.mean(background), np.std(background)

    # 4. CNR 公式: |mu_target - mu_background| / sqrt(sigma_target^2 + sigma_background^2)
    denominator = np.sqrt(sigma_t ** 2 + sigma_b ** 2)
    return np.abs(mu_t - mu_b) / denominator if denominator != 0 else 0