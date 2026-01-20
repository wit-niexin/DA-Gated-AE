"""
Module: Evaluation - US-DRUNet
"""

import os
import time
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from thop import profile

# --- 1. 环境与路径标准化 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import (
    ExperimentLogger,
    calculate_psnr, calculate_ssim, calculate_enl, calculate_cnr, calculate_rmse, calculate_epi
)
from models import get_model

# ==========================================
# 全局实验配置
# ==========================================
MODEL_NAME = "us_drunet"
SAVE_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, f"checkpoints/{MODEL_NAME}/{MODEL_NAME}_best.pth")
NOISE_LEVELS = [0.001, 0.02, 0.5]


def estimate_nakagami_map(img_tensor):
    """
    US-DRUNet 专用的统计参数估计函数
    """
    b, _, h, w = img_tensor.shape
    mu = torch.full((b, 1, h, w), 1.5).to(img_tensor.device)
    omega = torch.mean(img_tensor, dim=(2, 3), keepdim=True).expand(b, 1, h, w)
    return torch.cat([mu, omega], dim=1)


def run_us_drunet_experiment():
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", MODEL_NAME)
    qualitative_dir = os.path.join(results_dir, "qualitative")  # 补齐这个变量
    os.makedirs(denoised_base_dir, exist_ok=True)
    os.makedirs(qualitative_dir, exist_ok=True)

    # --- 3. 模型准备 ---
    model = get_model(MODEL_NAME).to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"✅ 已加载 {MODEL_NAME} 预训练权重。")
    model.eval()

    # --- 4. 计算复杂度 (US-DRUNet 特殊处理) ---
    # 由于 metrics.py 不支持双输入，我们这里单独算一次
    print("正在计算 US-DRUNet 复杂度 (Dual Input)...")
    input_size = (1, 1, 256, 256)
    dummy_img = torch.randn(input_size).to(DEVICE)
    dummy_nak = estimate_nakagami_map(dummy_img)

    params_count = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = 0.0
    if profile:
        # 特殊：inputs=(dummy_img, dummy_nak)
        flops, _ = profile(model, inputs=(dummy_img, dummy_nak), verbose=False)
        gflops = flops / 1e9
    print(f"Model Complexity -> Params: {params_count:.2f}M, GFLOPs: {gflops:.3f}")

    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=results_dir)
    results_summary = []

    # --- 5. 遍历噪声等级 ---
    for sigma_val in NOISE_LEVELS:
        print(f"\n🚀 [US-DRUNet] Sigma: {sigma_val}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")
        if not os.path.exists(noisy_dir): continue

        metrics_cache = {'psnr': [], 'ssim': [], 'rmse': [], 'epi': [], 'enl': [], 'cnr': [], 'time': []}
        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]

        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES: os.makedirs(save_path, exist_ok=True)

        # --- 6. 推理循环 ---
        for name in tqdm(img_names):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)
            if img_clean is None: continue

            # 预处理
            noisy_tensor = torch.from_numpy(img_noisy).float().div(255).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # A. 计时开始
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                start_t = time.time()

                # B. 实时生成先验 + 前向传播
                nak_map = estimate_nakagami_map(noisy_tensor)
                output_tensor = model(noisy_tensor, nak_map)

                # C. 计时结束
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                elapsed = time.time() - start_t

                # D. 后处理
                denoised = output_tensor.squeeze().cpu().clamp(0, 1).numpy() * 255
                denoised = denoised.astype(np.uint8)

            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised, save_path, qualitative_dir,
                                   img_names)

            # 指标计算
            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['rmse'].append(calculate_rmse(img_clean, denoised))
            metrics_cache['epi'].append(calculate_epi(denoised, img_clean))
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['time'].append(elapsed)

        # --- 7. 数据汇总 ---
        res = {
            "Noise": sigma_val,
            "PSNR": f"{np.mean(metrics_cache['psnr']):.2f} ± {np.std(metrics_cache['psnr']):.2f}",
            "SSIM": f"{np.mean(metrics_cache['ssim']):.4f} ± {np.std(metrics_cache['ssim']):.4f}",
            "RMSE": f"{np.mean(metrics_cache['rmse']):.2f} ± {np.std(metrics_cache['rmse']):.2f}",
            "EPI": f"{np.mean(metrics_cache['epi']):.4f} ± {np.std(metrics_cache['epi']):.4f}",
            "ENL": f"{np.mean(metrics_cache['enl']):.2f} ± {np.std(metrics_cache['enl']):.2f}",
            "CNR": f"{np.mean(metrics_cache['cnr']):.2f} ± {np.std(metrics_cache['cnr']):.2f}",
            "FPS": f"{1.0 / np.mean(metrics_cache['time']):.1f}",
            "GFLOPs": f"{gflops:.3f}",
            "Params": f"{params_count:.2f} M"
        }
        results_summary.append(res)

    df = pd.DataFrame(results_summary)
    logger.record_log(df)
    logger.save_csv(results_summary)
    print(f"\n✅ US-DRUNet 评估任务结束。")


if __name__ == "__main__":
    run_us_drunet_experiment()
