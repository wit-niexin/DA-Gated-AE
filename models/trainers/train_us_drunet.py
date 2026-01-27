"""
Module: Training Script - US-DRUNet
Description:
    US-DRUNet 的标准化训练脚本。

Key Adjustments for US-DRUNet:
    1. 双输入机制: 除了含噪图像，还需实时估计并输入 Nakagami 统计参数图。
    2. 混合损失对标: 使用与 DA-Gated AE 一致的 HybridLoss (1.0, 0.5, 0.1)。
    3. 随机噪声策略: 遵循“One Model for All”原则，在训练中随机合成三种强度的相关散斑噪声。
"""

import os
import sys
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 环境与路径标准化 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import DenoisingDataset, ExperimentLogger, calculate_psnr
from utils.losses import HybridLoss
from models import get_model

# ==========================================
# 训练超参数配置
# ==========================================
MODEL_NAME = "us_drunet"
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径管理
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, f"checkpoints/{MODEL_NAME}")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def estimate_nakagami_param(img_tensor):
    """
    【统计嵌入核心】
    根据输入图像实时估计 Nakagami 统计参数图 (mu, omega)。
    这是 US-DRUNet 区别于普通 CNN 的关键输入。
    """
    b, _, h, w = img_tensor.shape
    mu = torch.full((b, 1, h, w), 1.5).to(img_tensor.device)
    omega = torch.mean(img_tensor, dim=(2, 3), keepdim=True).expand(b, 1, h, w)
    return torch.cat([mu, omega], dim=1)


def validate(model, val_loader):
    """验证集评估逻辑：计算HybridLoss（与训练目标一致）"""
    model.eval()
    total_loss = 0
    count = 0
    criterion = HybridLoss(lambda_rec=1.0, lambda_ssim=0.5, lambda_edge=0.1).to(DEVICE)
    
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            # US-DRUNet 特有的辅助输入
            nak_map = estimate_nakagami_param(noisy)
            output = model(noisy, nak_map)
            loss = criterion(output, clean)
            total_loss += loss.item()
            count += 1
    
    return total_loss / count

# 原始PSNR验证函数（已注释，保留用于对比分析）
# def validate_psnr(model, val_loader):
#     model.eval()
#     psnr_list = []
#     with torch.no_grad():
#         for noisy, clean in val_loader:
#             noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
#             # US-DRUNet 特有的辅助输入
#             nak_map = estimate_nakagami_param(noisy)
#             output = model(noisy, nak_map)
#
#             clean_np = (clean.cpu().numpy() * 255).astype('uint8')
#             output_np = (torch.clamp(output, 0, 1).cpu().numpy() * 255).astype('uint8')
#             for i in range(clean_np.shape[0]):
#                 psnr = calculate_psnr(clean_np[i, 0], output_np[i, 0])
#                 psnr_list.append(psnr)
#     return np.mean(psnr_list)


def train():
    """
    US-DRUNet 核心训练流程
    """
    # --- 2. 初始化日志记录器 ---
    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=RESULTS_DIR)
    logger.log_text(f"🔔 [Baseline] 启动 {MODEL_NAME} 训练 | 策略: 统计嵌入 + 验证集考核")

    # --- 3. 数据准备 ---
    train_dataset = DenoisingDataset(
        clean_dir=os.path.join(PROJECT_ROOT, "data/train/clean"),
        noisy_root=os.path.join(PROJECT_ROOT, "data/train"),
        patch_size=256
    )
    val_dataset = DenoisingDataset(
        clean_dir=os.path.join(PROJECT_ROOT, "data/val/clean"),
        noisy_root=os.path.join(PROJECT_ROOT, "data/val"),
        patch_size=256
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 4. 架构与目标函数初始化 ---
    model = get_model(MODEL_NAME).to(DEVICE)
    criterion = HybridLoss(lambda_rec=1.0, lambda_ssim=0.5, lambda_edge=0.1).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 5. 训练核心循环 ---
    best_loss = float('inf')  # 【修改】改为基于loss保存（越小越好）
    patience_count = 0
    EARLY_STOP_PATIENCE = 30

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, leave=False)
        for batch_idx, (noisy, clean) in enumerate(loop):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            nak_map = estimate_nakagami_param(noisy)

            output = model(noisy, nak_map)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # --- 6. 验证与保存 (每 5 轮) ---
        if epoch % 5 == 0:
            val_loss = validate(model, val_loader)
            log_msg = f"Epoch {epoch}: Loss={avg_epoch_loss:.5f} | Val Loss={val_loss:.5f}"

            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth"))
                log_msg += " ⭐ (New Best!)"
            else:
                patience_count += 1

            print(log_msg)
            logger.log_text(log_msg)

            if patience_count >= EARLY_STOP_PATIENCE:
                break

        # --- 实验指标保存 ---
        logger.save_csv([{
            "epoch": epoch,
            "loss": f"{avg_epoch_loss:.6f}",
            "lr": f"{current_lr:.2e}",
            "best_loss": f"{best_loss:.5f}"
        }])

# 原始PSNR保存逻辑（已注释，保留用于对比分析）
# if epoch % 5 == 0:
#     val_psnr = validate_psnr(model, val_loader)
#     log_msg = f"Epoch {epoch}: Loss={avg_epoch_loss:.5f} | Val PSNR={val_psnr:.2f}dB"
#
#     if val_psnr > best_psnr:
#         best_psnr = val_psnr
#         patience_count = 0
#         torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth"))
#         log_msg += " ⭐ (New Best!)"
#     else:
#         patience_count += 1

    logger.log_text(f"🎉 {MODEL_NAME} 训练完成。最优 Loss: {best_loss:.5f}")


if __name__ == "__main__":
    train()