"""
Module: Baseline - Speckle Reducing Anisotropic Diffusion (SRAD)
Description:
    该脚本实现了 SRAD 算法，这是一种基于偏微分方程 (PDE) 的超声图像专用去噪算法。
    SRAD 改进了传统的 Perona-Malik 扩散模型，引入了针对散斑噪声统计特性的扩散系数。
    该模块能够有效抑制超声图像中的颗粒噪声，同时增强组织边缘。

Mathematical Logic:
    SRAD 通过迭代更新图像，每一代根据梯度和拉普拉斯算子计算扩散系数 c(q)。
    在同质区域（均匀组织），c(q) 趋近于 1，执行强力平滑；
    在异质区域（卵泡边缘），c(q) 趋近于 0，保持边缘不被模糊。
"""

import os
import time
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. 统一路径与环境配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 从 utils 门面接口导入工具
from utils import ExperimentLogger, calculate_psnr, calculate_ssim, calculate_enl, calculate_cnr

# ==========================================
# 全局实验配置
# ==========================================
SAVE_IMAGES = True
N_ITER = 30  # 迭代次数：超声论文常用 30-50 轮。越多则越平滑，但计算耗时增加。
DELTA_T = 0.05  # 步长：数值越小计算越稳定，数值越大去噪速度越快。
NOISE_LEVELS = [0.001, 0.02, 0.5]


def srad_filter(img, n_iter=30, delta_t=0.05):
    """
    SRAD 算法的 NumPy 高性能向量化实现。

    Args:
        img (ndarray): 输入灰度图像。
        n_iter (int): 迭代轮数。
        delta_t (float): 时间步长。
    Returns:
        ndarray: 去噪后的 8 位图像。
    """
    # 1. 转换为浮点型进行高精度偏微分计算
    I = img.astype(np.float32)

    # 初始散斑估计：根据迭代轮数预估对比度变异系数 q0
    q0 = np.exp(-n_iter * delta_t)

    for _ in range(n_iter):
        # 2. 计算四个方向的梯度 (利用 roll 进行矩阵位移操作，避免慢速的 for 循环)
        # dN = I(i-1, j) - I(i, j)
        dN = np.roll(I, -1, axis=0) - I
        dS = np.roll(I, 1, axis=0) - I
        dW = np.roll(I, -1, axis=1) - I
        dE = np.roll(I, 1, axis=1) - I

        # 3. 计算梯度的模长平方与拉普拉斯算子
        # 这是衡量图像局部特征的核心统计量
        grad_sq = (dN ** 2 + dS ** 2 + dW ** 2 + dE ** 2) / (I ** 2 + 1e-10)
        laplacian = (dN + dS + dW + dE) / (I + 1e-10)

        # 4. 计算瞬时散斑系数 q(t)
        # 基于偏微分方程的公式，用于定位噪声与边缘
        num = 0.5 * grad_sq - (1.0 / 16.0) * (laplacian ** 2)
        den = (1.0 + 0.25 * laplacian) ** 2
        q_sq = num / (den + 1e-10)

        # 5. 计算扩散系数 c(q)
        # 该系数决定了每个像素点的“扩散力度”
        c = 1.0 / (1.0 + (q_sq - q0 ** 2) / (q0 ** 2 * (1.0 + q0 ** 2) + 1e-10))
        c = np.clip(c, 0, 1)  # 保持数值稳定性

        # 6. 计算散度并更新图像
        # 通过扩散系数调整各方向的更新权重
        cS = np.roll(c, 1, axis=0)
        cE = np.roll(c, 1, axis=1)

        D = (c * dN + cS * dS + c * dW + cE * dE)
        I = I + (delta_t / 4.0) * D

    # 还原到 0-255 范围并转回 8 位图
    return np.clip(I, 0, 255).astype(np.uint8)


def run_srad_experiment():
    """
    SRAD 自动化评估主流程
    """
    # --- 2. 路径初始化 ---
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    qualitative_dir = os.path.join(results_dir, "qualitative")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", "srad")

    # 确保输出目录存在
    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    # 初始化 ExperimentLogger
    logger = ExperimentLogger(model_name="srad", root_dir=results_dir)
    results_summary = []

    # --- 3. 遍历噪声等级 ---
    for sigma_val in NOISE_LEVELS:
        print(f"\n🚀 [SRAD 评估中] 噪声等级: {sigma_val} | 迭代: {N_ITER}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")

        if not os.path.exists(noisy_dir):
            continue

        # 准备该等级的指标容器
        metrics_cache = {'psnr': [], 'ssim': [], 'enl': [], 'cnr': [], 'time': []}
        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]

        # 针对该噪声等级创建独立的保存子目录
        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES:
            os.makedirs(save_path, exist_ok=True)

        # --- 4. 核心推理循环 ---
        for name in tqdm(img_names, desc=f"SRAD Sigma {sigma_val}"):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)

            if img_clean is None or img_noisy is None:
                continue

            # 计时开始 (SRAD 的运算量通常是 IBF 的 10 倍以上)
            start_t = time.time()

            # 执行 SRAD 滤波
            denoised = srad_filter(img_noisy, n_iter=N_ITER, delta_t=DELTA_T)

            elapsed = time.time() - start_t

            # 使用公共 Logger 保存产出
            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised,
                                   save_path, qualitative_dir, img_names)

            # 使用公共 Metrics 计算指标
            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['time'].append(elapsed)

        # --- 5. 统计当前等级的数据 ---
        res = {
            "Noise": sigma_val,
            "PSNR": f"{np.mean(metrics_cache['psnr']):.2f} ± {np.std(metrics_cache['psnr']):.2f}",
            "SSIM": f"{np.mean(metrics_cache['ssim']):.4f} ± {np.std(metrics_cache['ssim']):.4f}",
            "ENL": f"{np.mean(metrics_cache['enl']):.2f} ± {np.std(metrics_cache['enl']):.2f}",
            "CNR": f"{np.mean(metrics_cache['cnr']):.2f} ± {np.std(metrics_cache['cnr']):.2f}",
            "FPS": f"{1.0 / np.mean(metrics_cache['time']):.1f}",
            "Params": "0 (PDE-based)"
        }
        results_summary.append(res)

    # --- 6. 生成报告 ---
    df = pd.DataFrame(results_summary)
    csv_path = logger.save_csv(results_summary)
    log_path = logger.record_log(df)

    print(f"\n✅ SRAD 实验完成！数据已记录至 {csv_path}")


if __name__ == "__main__":
    run_srad_experiment()