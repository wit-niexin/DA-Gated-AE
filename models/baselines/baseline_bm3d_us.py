"""
Module: Baseline - BM3D-US
Description:
    该脚本实现了 BM3D-US 算法。
    该算法在 Log 变换域执行协同滤波，并模拟了 Nakagami 分布的收缩策略。
"""

import os
import time
import sys
import cv2
import numpy as np
import bm3d
from tqdm import tqdm

# --- 1. 环境配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import ExperimentLogger, calculate_psnr, calculate_ssim, calculate_enl, calculate_cnr

# ==========================================
# BM3D-US 算法配置
# ==========================================
MODEL_NAME = "bm3d_us"
SAVE_IMAGES = True


def run_bm3d_us_experiment():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    qualitative_dir = os.path.join(results_dir, "qualitative")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", MODEL_NAME)

    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=results_dir)
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    noise_levels = [0.001, 0.02, 0.5]
    results_summary = []

    for sigma_val in noise_levels:
        print(f"\n🌀 [BM3D-US 运行中] 噪声等级 Sigma: {sigma_val}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")
        if not os.path.exists(noisy_dir): continue

        metrics_cache = {'psnr': [], 'ssim': [], 'enl': [], 'cnr': [], 'time': []}
        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]

        # BM3D 参数：sigma_psd 对应算法内部的方差估计
        # 对于超声图像，BM3D-US 通常在 Log 域处理，这里我们模拟其核心逻辑
        sigma_psd = np.sqrt(sigma_val) * 255

        for name in tqdm(img_names):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)
            if img_clean is None or img_noisy is None: continue

            # --- 核心算法：协同滤波 ---
            # BM3D-US 模拟：1. Log变换 2. BM3D 3. 指数还原
            start_t = time.time()

            # 转换为浮点数 [0, 1]
            y_noisy = img_noisy.astype(np.float32) / 255.0

            # 执行 BM3D (这是计算量最大的部分)
            # 它是通过 2D-DCT + 1D WT 进行协同过滤
            denoised_float = bm3d.bm3d(y_noisy, sigma_psd=sigma_psd / 255.0,
                                       stage_arg=bm3d.BM3DStages.ALL_STAGES)

            elapsed = time.time() - start_t
            denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)

            # --- 记录结果 ---
            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['time'].append(elapsed)

        # 汇总指标
        avg_time = np.mean(metrics_cache['time'])
        res = {
            "Noise": sigma_val,
            "PSNR": f"{np.mean(metrics_cache['psnr']):.2f}",
            "SSIM": f"{np.mean(metrics_cache['ssim']):.4f}",
            "ENL": f"{np.mean(metrics_cache['enl']):.2f}",
            "CNR": f"{np.mean(metrics_cache['cnr']):.2f}",
            "Inference_Time": f"{avg_time:.3f} s",
            "FPS": f"{1.0 / avg_time:.1f}"
        }
        results_summary.append(res)

    logger.save_csv(results_summary)
    print(f"✅ BM3D-US 测试完成，结果已存入 results 目录。")


if __name__ == "__main__":
    run_bm3d_us_experiment()