"""
Module: Metrics
Description:
    本模块集成了论文要求的 9 大评价指标，分为三类：
    1. 精度指标 (Accuracy): PSNR, SSIM, RMSE - 衡量去噪图像与 Ground Truth 的数值接近度。
    2. 物理指标 (Clinical/Physical): ENL, CNR, EPI - 评估超声特有的平滑度、对比度和边缘保真度。
    3. 效率指标 (Efficiency): Params, GFLOPs, FPS - 评估轻量化程度与实时处理能力。
"""

import numpy as np
import cv2
import torch
import time
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from scipy.ndimage import convolve
from thop import profile


# =================================================================
# 第一部分：图像质量指标 (Image Quality Metrics)
# =================================================================

def calculate_psnr(img_true, img_test, data_range=255):
    """
    计算峰值信噪比 (PSNR)。
    数学原理：PSNR = 10 * log10(MAX^2 / MSE)。
    意义：衡量重构像素值的整体精确度，分贝(dB)越高，重构效果越好。
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
    数学原理：综合考虑亮度(l)、对比度(c)和结构(s)三个分量。
    意义：比 PSNR 更符合人类视觉，SSIM 越接近 1，图像的纹理和解剖结构保持越完整。
    Args:
        img_true (ndarray): 参考图。
        img_test (ndarray): 待评价图。
    Returns:
        float: SSIM 值，范围 [0, 1]。
    """
    return ssim_func(img_true, img_test, data_range=data_range)


def calculate_rmse(img1, img2):
    """
    计算均方根误差 (RMSE)。
    数学原理：sqrt( mean( (img1 - img2)^2 ) )。
    意义：对图像中的大误差更敏感，反映去噪后的数值偏离程度。
    """
    return np.sqrt(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))


# =================================================================
# 第二部分：超声专用物理指标 (Ultrasound-specific Metrics)
# =================================================================

def calculate_enl(image, window_size=32):
    """
    计算等效视数 (Equivalent Number of Looks, ENL)。
    计算过程：
    1. 计算图像同质区域的均值 (μ) 和标准差 (σ)。
    2. 公式：ENL = (μ^2) / (σ^2)。
    意义：衡量对散斑噪声的平滑能力。ENL 越高，说明背景组织（如卵巢基质）越均匀，噪声抑制越强。
    """
    h, w = image.shape
    # 提取中心区域
    ch, cw = h // 2, w // 2
    wh, ww = window_size // 2, window_size // 2
    roi = image[ch - wh:ch + wh, cw - ww:cw + ww]

    mu = np.mean(roi)
    sigma = np.std(roi)
    return (mu ** 2) / (sigma ** 2) if sigma > 1e-6 else 0


def calculate_cnr(image):
    """
    计算对比度信噪比 (Contrast-to-Noise Ratio, CNR)。
    计算过程：
    1. 使用 Otsu 法自动寻找阈值，将图像分为目标区（卵泡/暗区）和背景区（基质/亮区）。
    2. 分别计算两区的均值(μt, μb)和标准差(σt, σb)。
    3. 公式：CNR = |μt - μb| / sqrt(σt^2 + σb^2)。
    意义：衡量目标与背景的可分辨性。CNR 越高，医生越容易识别卵泡边界。
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


def calculate_epi(img_denoised, img_clean):
    """
    计算边缘保持指数 (Edge Preservation Index, EPI)。
    计算过程：
    1. 使用拉普拉斯算子对去噪图和干净图进行高通滤波，提取边缘分量。
    2. 计算两个边缘矩阵的相关系数。
    意义：衡量算法在去噪的同时，保留高频细节（如卵泡壁）的能力。越接近 1 越完美。
    """
    laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge_d = convolve(img_denoised.astype(np.float32), laplacian_kernel)
    edge_c = convolve(img_clean.astype(np.float32), laplacian_kernel)

    # 计算相关系数
    num = np.sum((edge_d - np.mean(edge_d)) * (edge_c - np.mean(edge_c)))
    den = np.sqrt(np.sum((edge_d - np.mean(edge_d)) ** 2) * np.sum((edge_c - np.mean(edge_c)) ** 2))
    return num / den if den != 0 else 0


# =================================================================
# 第三部分：效率指标 (Efficiency Metrics)
# =================================================================

def calculate_model_complexity(model, input_size=(1, 1, 256, 256), device='cuda'):
    """
    计算模型复杂度：Params 与 GFLOPs。
    意义：用于证明本研究采用的深度可分离卷积 (DSC) 的轻量化优势。
    """
    # 如果不是 torch 模型（如 IBF, SRAD等）
    if not isinstance(model, torch.nn.Module):
        return 0.0, 0.0

    dummy_input = torch.randn(input_size).to(device)
    model.to(device).eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6

    # 强制确保 profile 存在
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
    except Exception as e:
        print(f"⚠️ 无法计算 GFLOPs: {e}")
        gflops = 0.0

    return params, gflops


def calculate_fps(model, input_size=(1, 1, 256, 256), device='cuda', iterations=100):
    """
    测试模型推理速度 (FPS)。
    计算过程：通过多次推理取平均时间，注意包含 GPU 预热环节以获得真实物理性能。
    """
    dummy_input = torch.randn(input_size).to(device)
    model.to(device).eval()

    # 预热 GPU
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    # 开始计时
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()  # 强制同步等待 GPU 完成
    end = time.time()

    fps = iterations / (end - start)
    return fps