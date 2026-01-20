"""
Module: Evaluation - DU-Net (Dilated U-Net)
Description:
    该脚本负责对训练好的 DU-Net 模型进行性能评估。
    DU-Net 结合了 U-Net 的跳跃连接（Skip Connections）和空洞卷积（Dilated Convolution），
    能够在不丢失空间分辨率的情况下扩大感受野，非常适合提取超声图像中复杂的散斑噪声分布。

Key Features:
    - 自动检测并利用 CUDA (GPU) 加速推理。
    - 包含完整的张量预处理与后处理逻辑。
    - 自动统计模型参数量 (Parameters)，用于论文中“模型复杂度”的对比。
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

from utils import (
    ExperimentLogger,
    calculate_psnr, calculate_ssim, calculate_enl, calculate_cnr, calculate_rmse, calculate_epi, calculate_model_complexity
)
from models import get_model

# ==========================================
# 全局实验配置
# ==========================================
SAVE_IMAGES = True
# 自动选择硬件：如果 AutoDL 有显卡则使用 GPU，否则退回 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 权重文件路径
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints/dunet/dunet_best.pth")
NOISE_LEVELS = [0.001, 0.02, 0.5]


def run_dunet_experiment():
    """
    DU-Net 自动化评估流程
    """
    # --- 2. 路径初始化 ---
    test_clean_dir = os.path.join(PROJECT_ROOT, "data/test/clean")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    qualitative_dir = os.path.join(results_dir, "qualitative")
    denoised_base_dir = os.path.join(results_dir, "denoised_images", "dunet")

    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    # --- 3. 模型准备 ---
    model = get_model("dunet").to(DEVICE)

    # 加载预训练权重
    if os.path.exists(WEIGHTS_PATH):
        print(f"✅ 成功加载权重: {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    else:
        print(f"⚠️ 警告: 未找到权重文件！将使用随机初始化的模型进行测试（仅用于代码自检）。")

    model.eval()  # 必须步骤：将模型设为评估模式，关闭 Dropout 和 BatchNormalization 的更新

    print("正在计算模型复杂度...")
    params_count, gflops = calculate_model_complexity(model, input_size=(1, 1, 256, 256), device=DEVICE)
    print(f"Model Complexity -> Params: {params_count:.2f}M, GFLOPs: {gflops:.3f}")

    # 初始化日志记录器
    logger = ExperimentLogger(model_name="dunet", root_dir=results_dir)
    results_summary = []

    # --- 4. 遍历噪声等级 ---
    for sigma_val in NOISE_LEVELS:
        print(f"\n🚀 [DU-Net 评估中] 噪声等级: {sigma_val} | 运行设备: {DEVICE}")
        noisy_dir = os.path.join(PROJECT_ROOT, f"data/test/noisy_{sigma_val}")

        if not os.path.exists(noisy_dir):
            continue

        metrics_cache = {'psnr': [], 'ssim': [], 'rmse': [], 'epi': [], 'enl': [], 'cnr': [], 'time': []}
        img_names = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]

        save_path = os.path.join(denoised_base_dir, f"sigma_{sigma_val}")
        if SAVE_IMAGES:
            os.makedirs(save_path, exist_ok=True)

        # --- 5. 推理循环 ---
        for name in tqdm(img_names, desc=f"DU-Net Sigma {sigma_val}"):
            img_clean = cv2.imread(os.path.join(test_clean_dir, name), 0)
            img_noisy = cv2.imread(os.path.join(noisy_dir, name), 0)
            if img_clean is None or img_noisy is None:
                continue

            # --- 预处理 (Preprocessing) ---
            # 1. 转为 Tensor  2. 归一化 [0,1]  3. 增加 Batch 和 Channel 维度 [1, 1, H, W]
            input_tensor = torch.from_numpy(img_noisy).float().div(255).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # --- 推理 (Inference) ---
            with torch.no_grad():  # 核心：推理时禁止梯度计算，显著降低显存占用
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                start_t = time.time()

                output_tensor = model(input_tensor)

                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                elapsed = time.time() - start_t

                # --- 后处理 (Post-processing) ---
                # 1. 降维  2. 截断异常值 [0,1]  3. 搬回 CPU  4. 恢复 0-255 并转为 uint8
                denoised = output_tensor.squeeze().cpu().clamp(0, 1).numpy() * 255
                denoised = denoised.astype(np.uint8)

            # --- 保存与指标计算 ---
            if SAVE_IMAGES:
                logger.save_images(name, sigma_val, img_clean, img_noisy, denoised, save_path, qualitative_dir, img_names)

            metrics_cache['psnr'].append(calculate_psnr(img_clean, denoised))
            metrics_cache['ssim'].append(calculate_ssim(img_clean, denoised))
            metrics_cache['rmse'].append(calculate_rmse(img_clean, denoised)) 
            metrics_cache['epi'].append(calculate_epi(denoised, img_clean))   
            metrics_cache['enl'].append(calculate_enl(denoised))
            metrics_cache['cnr'].append(calculate_cnr(denoised))
            metrics_cache['time'].append(elapsed)

        # --- 6. 数据汇总记录 ---
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

    # 生成最终报告
    df = pd.DataFrame(results_summary)
    logger.record_log(df)
    logger.save_csv(results_summary)
    print(f"\n✅ DU-Net 评估完成。指标已存入 results 文件夹。")


if __name__ == "__main__":
    run_dunet_experiment()
