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

from utils import (
    ExperimentLogger,
    calculate_psnr,
    calculate_ssim,
    calculate_rmse,
    calculate_enl,
    calculate_cnr,
    calculate_epi
)

# ==========================================
# 全局实验配置
# ==========================================
SAVE_IMAGES = True
N_ITER = 30  # 迭代次数
DELTA_T = 0.05  # 时间步长
NOISE_LEVELS = [0.001, 0.02, 0.5]


def srad_filter(img, n_iter=30, delta_t=0.05):
    """
    SRAD 算法的 NumPy 高性能向量化实现。
    """
    I = img.astype(np.float32)
    q0 = np.exp(-n_iter * delta_t)

    for _ in range(n_iter):
        # 计算梯度 (利用 roll 进行矩阵位移)
        dN = np.roll(I, -1, axis=0) - I
        dS = np.roll(I, 1, axis=0) - I
        dW = np.roll(I, -1, axis=1) - I
        dE = np.roll(I, 1, axis=1) - I

        # 计算局部特征统计量
        grad_sq = (dN ** 2 + dS ** 2 + dW ** 2 + dE ** 2) / (I ** 2 + 1e-10)
        laplacian = (dN + dS + dW + dE) / (I + 1e-10)

        # 计算扩散系数 c(q)
        num = 0.5 * grad_sq - (1.0 / 16.0) * (laplacian ** 2)
        den = (1.0 + 0.25 * laplacian) ** 2
        q_sq = num / (den + 1e-10)

        c = 1.0 / (1.0 + (q_sq - q0 ** 2) / (q0 ** 2 * (1.0 + q0 ** 2) + 1e-10))
        c = np.clip(c, 0, 1)

        # 更新图像
        cS = np.roll(c, 1, axis=0)
        cE = np.roll(c, 1, axis=1)
        D = (c * dN + cS * dS + c * dW + cE * dE)
        I = I + (delta_t / 4.0) * D

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

    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    logger = ExperimentLogger(model_name="srad", root_dir=results_dir)
    results_summary = []

    # --- 3. 遍历噪声等级 ---
    for sigma_val in NOISE_LEVELS:
        print(f"\n🚀 [SRAD 评估中] 噪声等级: {sigma_val} | 迭代: {N_ITER}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")

        if not os.path.exists(noisy_dir):
            print(f"⚠️ 警告: 找不到噪声目录 {noisy_dir}, 跳过此等级。")
            continue

        # 准备该等级的指标容器 (9大指标)
        metrics_cache = {
            'psnr': [], 'ssim': [], 'rmse': [],
            'enl': [], 'cnr': [], 'epi': [],
            'time': []
        }

        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]
        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES:
            os.makedirs(save_path, exist_ok=True)

        # --- 4. 核心推理循环 ---
        for name in tqdm(img_names, desc=f"SRAD Sigma {sigma_val}"):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)

            if img_clean is None or img_noisy is None:
                continue

            # 计时开始
            start_t = time.time()
            denoised = srad_filter(img_noisy, n_iter=N_ITER, delta_t=DELTA_T)
            elapsed = time.time() - start_t

            # 保存结果
            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised,
                                   save_path, qualitative_dir, img_names)

            # 指标计算
            # 精度
            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['rmse'].append(calculate_rmse(img_clean, denoised))
            # 物理
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['epi'].append(calculate_epi(denoised, img_clean))
            # 效率
            metrics_cache['time'].append(elapsed)

        # --- 5. 汇总当前等级数据 ---
        avg_time = np.mean(metrics_cache['time'])
        fps = 1.0 / avg_time if avg_time > 0 else 0

        res = {
            "Method": "SRAD",
            "Noise": sigma_val,
            # 精度
            "PSNR": f"{np.mean(metrics_cache['psnr']):.2f} ± {np.std(metrics_cache['psnr']):.2f}",
            "SSIM": f"{np.mean(metrics_cache['ssim']):.4f} ± {np.std(metrics_cache['ssim']):.4f}",
            "RMSE": f"{np.mean(metrics_cache['rmse']):.2f} ± {np.std(metrics_cache['rmse']):.2f}",
            # 物理
            "ENL": f"{np.mean(metrics_cache['enl']):.2f} ± {np.std(metrics_cache['enl']):.2f}",
            "CNR": f"{np.mean(metrics_cache['cnr']):.2f} ± {np.std(metrics_cache['cnr']):.2f}",
            "EPI": f"{np.mean(metrics_cache['epi']):.4f} ± {np.std(metrics_cache['epi']):.4f}",
            # 效率 (PDE 算法无 Params/GFLOPs)
            "Params(M)": "-",
            "GFLOPs": "-",
            "FPS": f"{fps:.1f}"
        }
        results_summary.append(res)

    # --- 6. 生成报告 ---
    df = pd.DataFrame(results_summary)
    cols = ["Method", "Noise", "PSNR", "SSIM", "RMSE", "ENL", "CNR", "EPI", "Params(M)", "GFLOPs", "FPS"]
    df = df[cols]

    csv_path = logger.save_csv(results_summary)
    log_path = logger.record_log(df)

    print(f"\n✅ SRAD 实验完成！报告已生成。")
    print(f"📊 CSV: {csv_path}")
    print(f"📝 Log: {log_path}")


if __name__ == "__main__":
    run_srad_experiment()
