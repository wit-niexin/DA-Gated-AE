"""
Module: Training Script - DU-Net (Baseline)
Description:
    基准模型 DU-Net 的标准化训练脚本。

Research Strategy:
    1. 统一损失函数 (Hybrid Loss):
       为了确保架构对比的公平性，本脚本放弃了原论文可能使用的简单 MSE，
       转而采用与 DA-Gated AE 完全一致的混合损失函数（Charbonnier + SSIM + Edge）。
       这确保了实验结论的差异性仅来源于网络拓扑结构。
    2. 实验追踪 (Experiment Tracking):
       引入 ExperimentLogger，确保基准模型的收敛曲线可追溯、可对比。
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 环境与路径标准化 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import DenoisingDataset, ExperimentLogger
from utils.losses import HybridLoss  # 导入相同的混合损失
from models import get_model

# ==========================================
# 训练超参数配置 (保持与 Proposed Method 一致)
# ==========================================
MODEL_NAME = "dunet"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径管理
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, f"checkpoints/{MODEL_NAME}")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    """
    DU-Net 核心训练流程
    """
    # --- 2. 初始化日志记录器 (科研数据支撑的关键) ---
    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=RESULTS_DIR)
    logger.record_log(f"🔔 [Baseline] 启动 {MODEL_NAME} 对标训练 | 设备: {DEVICE}")

    # --- 3. 数据准备 ---
    train_dataset = DenoisingDataset(
        clean_dir=os.path.join(PROJECT_ROOT, "data/train/clean"),
        noisy_root=os.path.join(PROJECT_ROOT, "data/train"),
        patch_size=256
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # --- 4. 架构与目标函数初始化 ---
    model = get_model(MODEL_NAME).to(DEVICE)

    # 使用混合损失函数代替单 MSELoss
    criterion = HybridLoss(
        lambda_rec=1.0,
        lambda_ssim=0.2,
        lambda_edge=0.5
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"🚀 实验对标启动 | 模型: {MODEL_NAME} | 目标函数: HybridLoss")

    # --- 5. 训练核心循环 ---
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, leave=False)
        for batch_idx, (noisy, clean) in enumerate(loop):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            # 前向传播 (Forward Pass)
            output = model(noisy)
            loss = criterion(output, clean)

            # 反向传播 (Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # 周期性汇总与记录
        avg_epoch_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # --- 6. 实验指标持久化 ---
        # 记录每轮数据至 CSV，用于后期绘制收敛对比图
        logger.save_csv([{
            "epoch": epoch,
            "loss": f"{avg_epoch_loss:.6f}",
            "lr": f"{current_lr:.2e}",
            "best_loss": f"{min(best_loss, avg_epoch_loss):.6f}"
        }])

        # 逻辑 A: 保存“最优权重” (供 Baseline 测试脚本读取)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), best_path)
            log_msg = f"⭐ Epoch {epoch}: 发现更优基准模型 (Loss: {avg_epoch_loss:.6f})"
        else:
            log_msg = f"Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}"

        # 将进度同时写入控制台和 TXT 日志
        if epoch % 1 == 0:  # 每一轮都记录日志
            logger.record_log(log_msg)

        # 逻辑 B: 每 10 轮定期归档
        if epoch % 10 == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 基准备份: {save_path}")

    logger.record_log(f"🎉 {MODEL_NAME} 训练任务圆满完成。最优损失: {best_loss:.6f}")


if __name__ == "__main__":
    train()