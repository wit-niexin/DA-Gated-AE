"""
Module: Training Script - DA-Gated AE (Proposed Method)
Description:
    该脚本实现了 DA-Gated AE 的静态数据集训练流程，以确保与 Baseline 模型在相同噪声样本下进行公平对比。
    保留了核心的 Hybrid Loss 和基于验证集 PSNR 的模型选择策略。
"""

import os
import sys
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 动态路径导入 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import DenoisingDataset, HybridLoss, ExperimentLogger, calculate_psnr
from models import get_model

# ==========================================
# 训练超参数配置 (严格对齐 Baseline)
# ==========================================
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "da_gated_ae"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, f"checkpoints/{MODEL_NAME}")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def validate(model, val_loader):
    """验证集评估逻辑：计算平均 PSNR"""
    model.eval()
    psnr_list = []
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            output = model(noisy)

            # 转为 Numpy 计算指标
            clean_np = (clean.cpu().numpy() * 255).astype('uint8')
            output_np = (torch.clamp(output, 0, 1).cpu().numpy() * 255).astype('uint8')

            for i in range(clean_np.shape[0]):
                psnr = calculate_psnr(clean_np[i, 0], output_np[i, 0])
                psnr_list.append(psnr)
    return np.mean(psnr_list)


def train():
    # --- 2. 数据载入 ---
    # 训练集：用于更新梯度
    train_dataset = DenoisingDataset(
        clean_dir=os.path.join(PROJECT_ROOT, "data/train/clean"),
        noisy_root=os.path.join(PROJECT_ROOT, "data/train"),
        patch_size=256
    )

    # 验证集：用于每 5 轮评估一次，保存 Best Model
    val_dataset = DenoisingDataset(
        clean_dir=os.path.join(PROJECT_ROOT, "data/val/clean"),
        noisy_root=os.path.join(PROJECT_ROOT, "data/val"),
        patch_size=256
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 3. 初始化 ---
    model = get_model(MODEL_NAME).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = HybridLoss(lambda_rec=1.0, lambda_ssim=0.5, lambda_edge=0.1).to(DEVICE)
    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=RESULTS_DIR)

    best_psnr = 0.0
    patience_count = 0
    EARLY_STOP_PATIENCE = 30

    print(f"🔥 {MODEL_NAME} 训练启动 (静态公平对比模式)")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}")

        for noisy, clean in loop:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            output = model(noisy)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = train_loss / len(train_loader)

        # --- 4. 验证与最优模型保存 (每 5 轮) ---
        if epoch % 5 == 0:
            val_psnr = validate(model, val_loader)
            print(f"Epoch {epoch}: Loss={avg_loss:.5f} | Val PSNR={val_psnr:.2f}dB")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                patience_count = 0
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth"))
                print(f"⭐ 发现新高！权重已保存至 {MODEL_NAME}_best.pth")
            else:
                patience_count += 1

            if patience_count >= EARLY_STOP_PATIENCE:
                print("🛑 触发早停。")
                break

        # 记录日志
        logger.save_csv([{"epoch": epoch, "loss": avg_loss, "best_psnr": best_psnr}])

    print(f"✅ 训练圆满完成。最高 PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    train()
