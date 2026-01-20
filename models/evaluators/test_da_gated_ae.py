"""
Module: Evaluation - DA-Gated AE (Proposed Method)
Description:
    该脚本用于评估论文提出的 DA-Gated AE 在测试集上的表现。
    它会加载训练好的权重，自动处理不同噪声等级的图像，
    并生成论文所需的定量结果 (CSV) 和定性对比图 (Images)。

Key Evaluation Metrics:
    - Quantitative: PSNR, SSIM (保真度), ENL, CNR (超声特化指标).
    - Efficiency: Model Size (MB), FPS (推理速度).
"""

import os
import time
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. 路径与环境配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import ExperimentLogger, calculate_psnr, calculate_ssim, calculate_enl, calculate_cnr
from models import get_model

# ==========================================
# 全局评估配置
# ==========================================
SAVE_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 指向你刚才训练脚本保存的权重路径
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints/da_gated_ae/da_gated_ae_best.pth")


def run_da_gated_ae_experiment():
    """
    DA-Gated AE 自动化测试流程
    """
    # --- 2. 路径初始化 ---
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    qualitative_dir = os.path.join(results_dir, "qualitative")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", "da_gated_ae")

    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    # --- 3. 模型准备与权重加载 ---
    model = get_model("da_gated_ae").to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        print(f"✅ 正在载入核心模型权重: {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    else:
        print(f"❌ 错误: 未找到权重文件 {WEIGHTS_PATH}。请先运行训练脚本。")
        return

    model.eval()

    # 统计模型参数量与体积 (对应论文 3.2.3 章节)
    params_count = sum(p.numel() for p in model.parameters()) / 1e6
    # 估算模型文件大小 (以 float32 计算)
    model_size_mb = (sum(p.numel() for p in model.parameters()) * 4) / (1024 * 1024)

    logger = ExperimentLogger(model_name="da_gated_ae", root_dir=results_dir)
    noise_levels = [0.001, 0.02, 0.5]
    results_summary = []

    # --- 4. 遍历噪声等级进行测试 ---
    for sigma_val in noise_levels:
        print(f"\n🚀 [DA-Gated AE 评估中] Sigma: {sigma_val} | 显存: {DEVICE}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")

        if not os.path.exists(noisy_dir): continue

        metrics_cache = {'psnr': [], 'ssim': [], 'enl': [], 'cnr': [], 'time': []}
        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]

        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES: os.makedirs(save_path, exist_ok=True)

        # --- 5. 核心推理循环 ---
        for name in tqdm(img_names, desc=f"DA-Gated AE Sigma {sigma_val}"):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)
            if img_clean is None or img_noisy is None: continue

            # 预处理
            input_tensor = torch.from_numpy(img_noisy).float().div(255).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # 推理
            start_t = time.time()
            with torch.no_grad():
                output_tensor = model(input_tensor)
                # 后处理
                denoised = output_tensor.squeeze().cpu().clamp(0, 1).numpy() * 255
                denoised = denoised.astype(np.uint8)

            elapsed = time.time() - start_t

            # 指标计算与存图
            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised,
                                   save_path, qualitative_dir, img_names)

            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['time'].append(elapsed)

        # --- 6. 汇总当前等级结果 ---
        avg_fps = 1.0 / np.mean(metrics_cache['time'])
        res = {
            "Noise": sigma_val,
            "PSNR": f"{np.mean(metrics_cache['psnr']):.2f} ± {np.std(metrics_cache['psnr']):.2f}",
            "SSIM": f"{np.mean(metrics_cache['ssim']):.4f} ± {np.std(metrics_cache['ssim']):.4f}",
            "ENL": f"{np.mean(metrics_cache['enl']):.2f} ± {np.std(metrics_cache['enl']):.2f}",
            "CNR": f"{np.mean(metrics_cache['cnr']):.2f} ± {np.std(metrics_cache['cnr']):.2f}",
            "FPS": f"{avg_fps:.1f}",
            "Params": f"{params_count:.2f} M ({model_size_mb:.2f} MB)"
        }
        results_summary.append(res)

    # 保存 CSV 和 Log
    logger.save_csv(results_summary)
    logger.record_log(pd.DataFrame(results_summary))
    print(f"\n✅ 评估完成！模型参数量: {params_count:.2f}M，建议查看 results 目录下的报告。")


if __name__ == "__main__":
    run_da_gated_ae_experiment()