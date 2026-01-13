"""
Module: Evaluation - Swin-US (Transformer-based Baseline)
Description:
    该脚本对 Swin-US 模型进行深度评估。
    Swin-US 利用移动窗口自注意力机制捕捉超声图像的长程依赖，
    但在推理速度和显存占用上通常高于 CNN 架构。

Research Rationale:
    1. 性能对标: 验证全局上下文建模对超声解剖结构（如卵泡边缘）的保护能力。
    2. 效率对标: 通过统计 FPS 和推理耗时，量化 Transformer 部署在便携设备上的挑战。
"""

import os
import time
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. 环境与路径标准化 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import ExperimentLogger, calculate_psnr, calculate_ssim, calculate_enl, calculate_cnr
from models import get_model

# ==========================================
# 全局实验配置
# ==========================================
MODEL_NAME = "swin_us"
SAVE_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, f"checkpoints/{MODEL_NAME}/{MODEL_NAME}_best.pth")
NOISE_LEVELS = [0.001, 0.02, 0.5]  # 对应低、中、高三种噪声环境


def run_swin_us_experiment():
    """
    Swin-US 自动化评估流程
    """
    # --- 2. 路径初始化 ---
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    qualitative_dir = os.path.join(results_dir, "qualitative")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", MODEL_NAME)

    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    # --- 3. 模型准备 ---
    model = get_model(MODEL_NAME).to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        print(f"✅ 成功加载 Swin-US 权重: {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    else:
        print(f"⚠️ 警告: 未找到 {MODEL_NAME} 权重文件！将使用随机初始化进行性能预估。")

    model.eval()

    # 统计参数量
    params_count = sum(p.numel() for p in model.parameters()) / 1e6

    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=results_dir)
    results_summary = []

    # --- 4. 遍历噪声等级进行测试 ---
    for sigma_val in NOISE_LEVELS:
        print(
            f"\n🌀 [Swin-US 评估中] Sigma: {sigma_val} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")

        if not os.path.exists(noisy_dir):
            print(f"❌ 跳过: 找不到目录 {noisy_dir}")
            continue

        metrics_cache = {'psnr': [], 'ssim': [], 'enl': [], 'cnr': [], 'time': []}
        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]

        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES: os.makedirs(save_path, exist_ok=True)

        # --- 5. 执行推理 ---
        for name in tqdm(img_names, desc=f"{MODEL_NAME.upper()} Processing"):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)
            if img_clean is None or img_noisy is None: continue

            # 图像预处理
            input_tensor = torch.from_numpy(img_noisy).float().div(255).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # 推理性能统计
            start_t = time.time()
            with torch.no_grad():
                # Transformer 架构前向传播
                output_tensor = model(input_tensor)

                # 同步 GPU 确保时间统计准确
                if DEVICE.type == 'cuda': torch.cuda.synchronize()

                denoised = output_tensor.squeeze().cpu().clamp(0, 1).numpy() * 255
                denoised = denoised.astype(np.uint8)

            elapsed = time.time() - start_t

            # 计算与保存指标
            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised,
                                   save_path, qualitative_dir, img_names)

            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['time'].append(elapsed)

        # --- 6. 生成汇总统计数据 (Table III 核心来源) ---
        avg_time_ms = np.mean(metrics_cache['time']) * 1000
        res = {
            "Noise": sigma_val,
            "PSNR": f"{np.mean(metrics_cache['psnr']):.2f} ± {np.std(metrics_cache['psnr']):.2f}",
            "SSIM": f"{np.mean(metrics_cache['ssim']):.4f} ± {np.std(metrics_cache['ssim']):.4f}",
            "ENL": f"{np.mean(metrics_cache['enl']):.2f}",
            "CNR": f"{np.mean(metrics_cache['cnr']):.2f}",
            "Latency": f"{avg_time_ms:.2f} ms",
            "FPS": f"{1000.0 / avg_time_ms:.1f}",
            "Params": f"{params_count:.2f} M"
        }
        results_summary.append(res)

    # 记录结果
    df = pd.DataFrame(results_summary)
    logger.record_log(f"\n--- {MODEL_NAME.upper()} Evaluation Results ---")
    logger.record_log(df)
    logger.save_csv(results_summary)
    print(f"\n✅ Swin-US 评估任务圆满完成。结果已同步至：results/{MODEL_NAME}")


if __name__ == "__main__":
    run_swin_us_experiment()