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
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 环境与路径标准化 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import DenoisingDataset, ExperimentLogger
from utils.losses import HybridLoss
from models import get_model

# ==========================================
# 训练超参数配置
# ==========================================
MODEL_NAME = "us_drunet"
BATCH_SIZE = 8  # 由于 US-DRUNet 参数量大，如果显存不足可适当调小
EPOCHS = 100
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
    # 在实际科研代码中，这里通常通过局部滑动窗口计算
    # 为保持 Demo 逻辑清晰，我们生成与输入尺寸一致的参数占位图
    # 维度: [B, 2, H, W] -> 通道 0 是 mu, 通道 1 是 omega
    b, _, h, w = img_tensor.shape
    mu = torch.full((b, 1, h, w), 1.5).to(img_tensor.device)      # 典型超声 mu 值
    omega = torch.mean(img_tensor, dim=(2, 3), keepdim=True).expand(b, 1, h, w)
    return torch.cat([mu, omega], dim=1)


def train():
    """
    US-DRUNet 核心训练流程
    """
    # --- 2. 初始化日志记录器 ---
    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=RESULTS_DIR)
    logger.log_text(f"🔔 [Baseline] 启动 {MODEL_NAME} 训练 | 策略: 统计嵌入 + 随机噪声")

    # --- 3. 数据准备 (使用我们决定的随机混合噪声逻辑) ---
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
    # 这里的 get_model("us_drunet") 应该返回我们之前写的 USDRUNet 类
    model = get_model(MODEL_NAME).to(DEVICE)

    # 【修正】严格对标论文 4.1.2 节确定的最佳权重：1.0, 0.5, 0.1
    criterion = HybridLoss(
        lambda_rec=1.0,
        lambda_ssim=0.5,
        lambda_edge=0.1
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 5. 训练核心循环 ---
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, leave=False)
        for batch_idx, (noisy, clean) in enumerate(loop):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            # Step A: 准备统计先验输入 (Nakagami Map)
            nak_map = estimate_nakagami_param(noisy)

            # Step B: 前向传播 (传递两个参数)
            output = model(noisy, nak_map)
            loss = criterion(output, clean)

            # Step C: 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # 汇总与记录
        avg_epoch_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # --- 6. 实验指标保存 ---
        logger.save_csv([{
            "epoch": epoch,
            "loss": f"{avg_epoch_loss:.6f}",
            "lr": f"{current_lr:.2e}"
        }])

        # 保存最优基准模型权重
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), best_path)
            log_msg = f"⭐ Epoch {epoch}: 发现更优 US-DRUNet (Loss: {avg_epoch_loss:.6f})"
        else:
            log_msg = f"Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}"

        logger.log_text(log_msg)

    logger.log_text(f"🎉 {MODEL_NAME} 训练完成。最优损失: {best_loss:.6f}")


if __name__ == "__main__":
    train()