"""
Module: Evaluation - DA-Gated AE (Proposed Method)
Description:
    该脚本用于评估论文提出的 DA-Gated AE 在测试集上的表现。
    为了确保公平性，评估流程、预处理逻辑及 9 大评价指标均与 Baseline 严格对齐。

Key Evaluation Metrics (9 Metrics):
    1-2. Fidelity: PSNR, SSIM
    3. Error: RMSE
    4. Structure: EPI (Edge Preservation Index)
    5. 6. Ultrasound Specific: ENL, CNR
    7-9. Efficiency: FPS, Params, GFLOPs
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

from utils import (
    ExperimentLogger, calculate_psnr, calculate_ssim,
    calculate_enl, calculate_cnr, calculate_rmse,
    calculate_epi, calculate_model_complexity
)
from models import get_model

# ==========================================
# 全局评估配置
# ==========================================
SAVE_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 指向训练脚本保存的最优权重路径
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints/da_gated_ae/da_gated_ae_best.pth")
NOISE_LEVELS = [0.001, 0.02, 0.5]


def run_da_gated_ae_experiment():
    """
    DA-Gated AE 自动化测试流程 (9大指标同步版)
    """
    # --- 2. 路径初始化 ---
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    qualitative_dir = os.path.join(results_dir, "qualitative")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", "da_gated_ae")

    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    # --- 3. 模型准备与复杂度分析 ---
    model = get_model("da_gated_ae").to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        print(f"✅ 成功加载 DA-Gated AE 最优权重: {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    else:
        print(f"❌ 错误: 未找到权重文件 {WEIGHTS_PATH}。请先运行训练脚本。")
        return

    model.eval()

    # 标准化计算模型复杂度 (Params & GFLOPs)
    print("正在分析模型复杂度 (GFLOPs)...")
    params_count, gflops = calculate_model_complexity(model, input_size=(1, 1, 256, 256), device=DEVICE)
    print(f"Proposed Model -> Params: {params_count:.2f}M, GFLOPs: {gflops:.3f}")

    logger = ExperimentLogger(model_name="da_gated_ae", root_dir=results_dir)
    results_summary = []

    # --- 4. 遍历噪声等级进行测试 ---
    for sigma_val in NOISE_LEVELS:
        print(f"\n🚀 [DA-Gated AE 评估中] 噪声等级: {sigma_val} | 运行设备: {DEVICE}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")

        if not os.path.exists(noisy_dir):
            print(f"跳过 Sigma {sigma_val}: 找不到数据目录 {noisy_dir}")
            continue

        # 初始化当前等级的缓存
        metrics_cache = {
            'psnr': [], 'ssim': [], 'rmse': [],
            'epi': [], 'enl': [], 'cnr': [], 'time': []
        }
        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]

        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES:
            os.makedirs(save_path, exist_ok=True)

        # --- 5. 核心推理循环 ---
        for name in tqdm(img_names, desc=f"DA-Gated AE Sigma {sigma_val}"):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)
            if img_clean is None or img_noisy is None:
                continue

            # --- 预处理 ---
            input_tensor = torch.from_numpy(img_noisy).float().div(255).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # --- 精准推理计时 ---
            with torch.no_grad():
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                start_t = time.time()

                output_tensor = model(input_tensor)

                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                elapsed = time.time() - start_t

                # --- 后处理 ---
                denoised = output_tensor.squeeze().cpu().clamp(0, 1).numpy() * 255
                denoised = denoised.astype(np.uint8)

            # --- 保存与 9 大指标计算 ---
            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised,
                                   save_path, qualitative_dir, img_names)

            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['rmse'].append(calculate_rmse(img_clean, denoised))
            metrics_cache['epi'].append(calculate_epi(denoised, img_clean))
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['time'].append(elapsed)

        # --- 6. 汇总当前等级结果 (Mean ± Std) ---
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

    # --- 7. 生成最终报告 ---
    df = pd.DataFrame(results_summary)
    logger.record_log(df)  # 记录到文本日志
    logger.save_csv(results_summary)  # 保存为 CSV 供 Excel 绘图

    print(f"\n✅ DA-Gated AE 评估圆满完成！")
    print(f"📊 最终平均 FPS: {df['FPS'].iloc[-1]} | 参数量: {params_count:.2f}M | 计算量: {gflops:.3f} GFLOPs")


if __name__ == "__main__":
    run_da_gated_ae_experiment()