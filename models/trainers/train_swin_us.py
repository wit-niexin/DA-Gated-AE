"""
Module: Training Script - Swin-US (Transformer-based Baseline)
Description:
    该脚本用于训练 Swin-US 基准模型。
    Swin-US 代表了基于全局上下文建模（Global Context Modeling）的先进去噪方法。

Detailed Comments & Research Rationale:
    1. 架构逻辑 : 使用 Swin Transformer 块捕捉长程依赖，处理超声图像中的大尺度结构。
    2. 损失函数 : 同样采用 Hybrid Loss，以确保在 PSNR/SSIM 指标上的公平竞争。
    3. 训练难点: Transformer 架构参数量通常较大且计算复杂。为了防止过拟合，使用了微调后的权重衰减。
"""

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # ✅ 新增：混合精度支持

# --- 1. 动态路径导入 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import DenoisingDataset, ExperimentLogger, calculate_psnr
from utils.losses import HybridLoss
from models import get_model

# ==========================================
# 训练配置 (针对显存深度优化版)
# ==========================================
MODEL_NAME = "swin_us"
BATCH_SIZE = 16
ACCUM_ITER = 4  # ✅ 新增：梯度累加步数，实际等效 Batch Size = 1 * 4 = 4
EPOCHS = 300  # Transformer 建议至少 100 轮
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径管理
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", MODEL_NAME)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 显存基础优化
torch.backends.cudnn.benchmark = True
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()


def validate(model, val_loader):
    """验证集评估逻辑：计算HybridLoss（与训练目标一致）"""
    model.eval()
    total_loss = 0
    count = 0
    criterion = HybridLoss(lambda_rec=1.0, lambda_ssim=0.5, lambda_edge=0.1).to(DEVICE)
    
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            # Transformer 通常建议关闭 autocast 进行验证以保证精度
            output = model(noisy)
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
#             # Transformer 通常建议关闭 autocast 进行验证以保证精度
#             output = model(noisy)
#             clean_np = (clean.cpu().numpy() * 255).astype('uint8')
#             output_np = (torch.clamp(output, 0, 1).cpu().numpy() * 255).astype('uint8')
#             for i in range(clean_np.shape[0]):
#                 psnr = calculate_psnr(clean_np[i, 0], output_np[i, 0])
#                 psnr_list.append(psnr)
#     return np.mean(psnr_list)


def train():
    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=RESULTS_DIR)
    logger.log_text(f"🚀 [Optimized] 启动 {MODEL_NAME} 训练 | 设备: {DEVICE}")

    # --- 数据加载 ---
    train_dataset = DenoisingDataset(
        clean_dir=os.path.join(PROJECT_ROOT, "data/train/clean"),
        noisy_root=os.path.join(PROJECT_ROOT, "data/train"),
        patch_size=128
    )
    val_dataset = DenoisingDataset(
        clean_dir=os.path.join(PROJECT_ROOT, "data/val/clean"),
        noisy_root=os.path.join(PROJECT_ROOT, "data/val"),
        patch_size=128
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 模型与损失函数 ---
    model = get_model(MODEL_NAME).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.log_text(f"📊 Swin-US 参数量: {total_params:.2f} M")

    criterion = HybridLoss(lambda_rec=1.0, lambda_ssim=0.5, lambda_edge=0.1).to(DEVICE)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ✅ 新增：初始化混合精度缩放器
    scaler = GradScaler()

    # --- 训练循环 ---
    best_loss = float('inf')  # 【修改】改为基于loss保存（越小越好）
    patience_count = 0
    EARLY_STOP_PATIENCE = 30

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        loop = tqdm(train_loader, leave=False)
        for batch_idx, (noisy, clean) in enumerate(loop):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            # ✅ 优化：使用 autocast 进行混合精度前向传播
            with autocast():
                output = model(noisy)
                loss = criterion(output, clean)
                loss = loss / ACCUM_ITER

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUM_ITER == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM_ITER
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item() * ACCUM_ITER)

        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # --- 验证与保存 (每 5 轮) ---
        if epoch % 5 == 0:
            val_loss = validate(model, val_loader)
            log_msg = f"Epoch {epoch}: Loss={avg_loss:.6f} | Val Loss={val_loss:.5f}"

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

        # --- 记录 ---
        logger.save_csv([{
            "epoch": epoch,
            "loss": f"{avg_loss:.6f}",
            "lr": f"{current_lr:.2e}",
            "best_loss": f"{best_loss:.5f}"
        }])

# 原始PSNR保存逻辑（已注释，保留用于对比分析）
# if epoch % 5 == 0:
#     val_psnr = validate_psnr(model, val_loader)
#     log_msg = f"Epoch {epoch}: Loss={avg_loss:.6f} | Val PSNR={val_psnr:.2f}dB"
#
#     if val_psnr > best_psnr:
#         best_psnr = val_psnr
#         patience_count = 0
#         torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth"))
#         log_msg += " ⭐ (New Best!)"
#     else:
#         patience_count += 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_epoch_{epoch}.pth"))

    logger.log_text(f"✨ 训练完成。最优 Loss: {best_loss:.5f}")


if __name__ == "__main__":
    train()