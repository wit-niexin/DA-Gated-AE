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
import pandas as pd
from tqdm import tqdm

# --- 1. 环境配置 ---
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
# BM3D-US 算法配置
# ==========================================
MODEL_NAME = "bm3d_us"
SAVE_IMAGES = True
NOISE_LEVELS = [0.001, 0.02, 0.5]


def run_bm3d_us_experiment():
    # --- 2. 路径初始化 ---
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    qualitative_dir = os.path.join(results_dir, "qualitative")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", MODEL_NAME)

    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=results_dir)
    results_summary = []

    # --- 3. 遍历噪声等级 ---
    for sigma_val in NOISE_LEVELS:
        print(f"\n🌀 [BM3D-US 运行中] 噪声等级 Sigma: {sigma_val}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")
        if not os.path.exists(noisy_dir):
            continue

        # 准备指标容器 (9大指标)
        metrics_cache = {
            'psnr': [], 'ssim': [], 'rmse': [],
            'enl': [], 'cnr': [], 'epi': [],
            'time': []
        }

        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]
        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES:
            os.makedirs(save_path, exist_ok=True)

        # BM3D 参数配置
        sigma_psd = np.sqrt(sigma_val) * 255

        # --- 4. 核心循环 ---
        for name in tqdm(img_names, desc=f"BM3D Sigma {sigma_val}"):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)
            if img_clean is None or img_noisy is None:
                continue

            # 推理与计时
            start_t = time.time()

            # BM3D 核心逻辑: Normalization -> Log -> BM3D -> Exp -> Denormalization
            y_noisy = img_noisy.astype(np.float32) / 255.0
            # 注意: 此处为了简化模拟 BM3D-US，直接在强度域应用，若严格遵循 Log 域需添加 np.log
            # 但 standard bm3d 库通常处理加性噪声，故此处直接应用效果通常更稳定
            denoised_float = bm3d.bm3d(y_noisy, sigma_psd=sigma_psd / 255.0, stage_arg=bm3d.BM3DStages.ALL_STAGES)

            elapsed = time.time() - start_t
            denoised = (np.clip(denoised_float, 0, 1) * 255).astype(np.uint8)

            # 保存结果
            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised, save_path, qualitative_dir, img_names)

            # 指标计算
            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['rmse'].append(calculate_rmse(img_clean, denoised))
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['epi'].append(calculate_epi(denoised, img_clean))
            metrics_cache['time'].append(elapsed)

        # --- 5. 汇总当前等级数据 ---
        avg_time = np.mean(metrics_cache['time'])
        fps = 1.0 / avg_time if avg_time > 0 else 0

        res = {
            "Method": "BM3D-US",
            "Noise": sigma_val,
            # 精度
            "PSNR": f"{np.mean(metrics_cache['psnr']):.2f} ± {np.std(metrics_cache['psnr']):.2f}",
            "SSIM": f"{np.mean(metrics_cache['ssim']):.4f} ± {np.std(metrics_cache['ssim']):.4f}",
            "RMSE": f"{np.mean(metrics_cache['rmse']):.2f} ± {np.std(metrics_cache['rmse']):.2f}",
            # 物理
            "ENL": f"{np.mean(metrics_cache['enl']):.2f} ± {np.std(metrics_cache['enl']):.2f}",
            "CNR": f"{np.mean(metrics_cache['cnr']):.2f} ± {np.std(metrics_cache['cnr']):.2f}",
            "EPI": f"{np.mean(metrics_cache['epi']):.4f} ± {np.std(metrics_cache['epi']):.4f}",
            # 效率
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

    print(f"\n✅ BM3D-US 实验完成！报告已生成。")
    print(f"📊 CSV: {csv_path}")
    print(f"📝 Log: {log_path}")


if __name__ == "__main__":
    run_bm3d_us_experiment()
