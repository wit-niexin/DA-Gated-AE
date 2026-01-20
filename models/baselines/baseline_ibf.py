"""
Module: Baseline - Iterative Bilateral Filter (IBF)
Description:
    该脚本实现了基于双边滤波（Bilateral Filter）的传统去噪基准测试。
    双边滤波是一种非线性滤波器，能够通过结合空域近邻度和值域相似度，在平滑噪声的同时保留超声图像的边缘细节。

Algorithm Note:
    Bilateral Filter 核心参数：
    - d: 过滤过程中使用的每个像素邻域的直径。
    - sigmaColor: 颜色空间滤波器 sigma，越大意味着邻域内越宽的颜色会被混合。
    - sigmaSpace: 坐标空间滤波器 sigma，越大意味着越远的像素会相互影响。
"""

import os
import time
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. 统一路径与环境配置 ---
# 获取项目根目录，确保脚本在任何位置运行都能正确导入 utils
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
    calculate_epi,
    # calculate_model_complexity (传统算法不需要计算模型复杂度，直接置零)
)

# ==========================================
# 全局配置
# ==========================================
SAVE_IMAGES = True  # 是否保存去噪后的 .jpg 图像
NOISE_LEVELS = [0.001, 0.02, 0.5]  # 测试的噪声方差等级


def run_ibf_experiment():
    """
    主实验流程：读取测试集 -> 传统算法去噪 -> 9大指标计算 -> 自动汇总报告
    """
    # --- 2. 路径初始化 ---
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    qualitative_dir = os.path.join(results_dir, "qualitative")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", "ibf")

    # 确保输出目录存在
    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    # 初始化实验记录器
    logger = ExperimentLogger(model_name="ibf", root_dir=results_dir)
    results_summary = []

    # --- 3. 遍历不同强度的噪声集 ---
    for sigma_val in NOISE_LEVELS:
        print(f"\n🚀 [IBF 评估中] 正在处理噪声等级: {sigma_val}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")

        if not os.path.exists(noisy_dir):
            print(f"⚠️ 警告: 找不到噪声目录 {noisy_dir}, 跳过此等级。")
            continue

        metrics_cache = {
            'psnr': [], 'ssim': [], 'rmse': [],
            'enl': [], 'cnr': [], 'epi': [],
            'time': []
        }

        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]

        # 针对该噪声等级创建独立的保存子目录
        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES:
            os.makedirs(save_path, exist_ok=True)

        # --- 4. 遍历单张图片进行去噪测试 ---
        for name in tqdm(img_names, desc=f"Sigma {sigma_val}"):
            # 读取图像 (灰度模式)
            # 注意: OpenCV 读取为 uint8 [0, 255]
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)

            if img_clean is None or img_noisy is None:
                continue

            # 算法推理与计时 (传统算法使用 CPU 时间)
            start_t = time.time()

            # --- IBF 算法调用 ---
            # 参数设置：d=5 (邻域直径), sigmaColor=25, sigmaSpace=25
            # 这些参数是基于经验值设置的，旨在平衡去噪与保边
            denoised = cv2.bilateralFilter(img_noisy, d=5, sigmaColor=25, sigmaSpace=25)

            elapsed = time.time() - start_t

            # 保存结果与生成论文对比图
            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised, save_path, qualitative_dir, img_names)

            # --- 统计指标计算 ---
            # 1. 精度指标
            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['rmse'].append(calculate_rmse(img_clean, denoised))

            # 2. 物理指标
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['epi'].append(calculate_epi(denoised, img_clean))

            # 3. 效率指标
            metrics_cache['time'].append(elapsed)

        # --- 5. 汇总当前噪声等级的数据 ---
        # 计算平均 FPS
        avg_time = np.mean(metrics_cache['time'])
        fps = 1.0 / avg_time if avg_time > 0 else 0

        res = {
            "Method": "IBF",
            "Noise": sigma_val,
            # 精度指标
            "PSNR": f"{np.mean(metrics_cache['psnr']):.2f} ± {np.std(metrics_cache['psnr']):.2f}",
            "SSIM": f"{np.mean(metrics_cache['ssim']):.4f} ± {np.std(metrics_cache['ssim']):.4f}",
            "RMSE": f"{np.mean(metrics_cache['rmse']):.2f} ± {np.std(metrics_cache['rmse']):.2f}",
            # 物理指标
            "ENL": f"{np.mean(metrics_cache['enl']):.2f} ± {np.std(metrics_cache['enl']):.2f}",
            "CNR": f"{np.mean(metrics_cache['cnr']):.2f} ± {np.std(metrics_cache['cnr']):.2f}",
            "EPI": f"{np.mean(metrics_cache['epi']):.4f} ± {np.std(metrics_cache['epi']):.4f}",
            # 效率指标 (传统算法 Params/GFLOPs 为 0 或 N/A)
            "Params(M)": "-",  # 传统算法无参数
            "GFLOPs": "-",  # 传统算法无法用 thop 计算
            "FPS": f"{fps:.1f}"
        }
        results_summary.append(res)

    # --- 6. 生成实验最终报告 ---
    df = pd.DataFrame(results_summary)

    cols = ["Method", "Noise", "PSNR", "SSIM", "RMSE", "ENL", "CNR", "EPI", "Params(M)", "GFLOPs", "FPS"]
    df = df[cols]

    csv_path = logger.save_csv(results_summary)
    log_path = logger.record_log(df)

    print(f"\n✅ IBF 实验结束！报告已生成。")
    print(f"📊 CSV 数据: {csv_path}")
    print(f"📝 历史日志: {log_path}")


if __name__ == "__main__":
    run_ibf_experiment()
