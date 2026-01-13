"""
Module: Training Script - Swin-US (Transformer-based Baseline)
Description:
    该脚本用于训练 Swin-US 基准模型。
    Swin-US 代表了基于全局上下文建模（Global Context Modeling）的先进去噪方法。

Detailed Comments & Research Rationale:
    1. 架构逻辑 (Section 2.3.3): 使用 Swin Transformer 块捕捉长程依赖，处理超声图像中的大尺度结构。
    2. 损失函数 (Section 3.4): 同样采用 Hybrid Loss，以确保在 PSNR/SSIM 指标上的公平竞争。
    3. 训练难点: Transformer 架构参数量通常较大且计算复杂。为了防止过拟合，使用了微调后的权重衰减。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 动态路径导入 ---
# 确保脚本能找到项目根目录下的 utils 和 models 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import DenoisingDataset, ExperimentLogger
from utils.losses import HybridLoss
from models import get_model

# ==========================================
# 训练配置 (针对 Transformer 优化的超参数)
# ==========================================
MODEL_NAME = "swin_us"
BATCH_SIZE = 8  # 因为 Transformer 显存占用较高，若 16 报错请保持 8
EPOCHS = 100  # 建议跑满 100 轮，Transformer 收敛通常比 CNN 慢
LR = 5e-5  # Transformer 建议使用更小的初始学习率，以防梯度爆炸
WEIGHT_DECAY = 1e-4  # 增加权重衰减，防止 Transformer 在小样本超声数据集上过拟合
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径管理：确保权重和日志分别存放
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", MODEL_NAME)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    """
    Swin-US 核心训练逻辑
    """
    # --- 2. 初始化实验记录器 ---
    # 记录训练日志、Loss 曲线数据等，用于论文 Figure 绘图
    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=RESULTS_DIR)
    logger.record_log(f"🚀 [Baseline] 启动 {MODEL_NAME} 训练 | 设备: {DEVICE}")

    # --- 3. 加载超声去噪数据集 ---
    # 采用 256x256 Patch 训练，这是医学图像处理的常用方案
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

    # --- 4. 初始化 Swin-US 模型与混合损失函数 ---
    # 按照论文 2.3.3 节所述，Swin-US 集成了自注意力机制
    model = get_model(MODEL_NAME).to(DEVICE)

    # 打印参数量，你会发现它显著高于你的 DA-Gated AE (1.7M)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.record_log(f"📊 Swin-US 参数量: {total_params:.2f} M (预计远超轻量级模型)")

    # 为了公平对比，所有深度学习模型均使用相同的混合损失 (Eq. 9-13)
    criterion = HybridLoss(
        lambda_rec=1.0,
        lambda_ssim=0.2,
        lambda_edge=0.5
    ).to(DEVICE)

    # 优化器：AdamW 通常对 Transformer 类模型效果更好
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 学习率调度：每 30 轮衰减 50%
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # --- 5. 训练循环 ---
    best_loss = float('inf')  # 初始化历史最低 Loss 为无穷大

    for epoch in range(1, EPOCHS + 1):
        model.train()  # 确保进入训练模式 (启用 DropPath 和 BatchNorm)
        epoch_loss = 0

        # 使用 tqdm 在控制台显示实时进度
        loop = tqdm(train_loader, leave=False)
        for batch_idx, (noisy, clean) in enumerate(loop):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            # 前向传播：Swin-US 会通过自注意力层计算全局依赖
            output = model(noisy)
            loss = criterion(output, clean)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # 计算本轮平均损失并记录学习率
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # --- 6. 实验记录与最优权重保存 ---
        # 保存到 CSV，方便之后用 Excel/Matplotlib 分析收敛性
        logger.save_csv([{
            "epoch": epoch,
            "loss": f"{avg_loss:.6f}",
            "lr": f"{current_lr:.2e}",
            "best_loss": f"{min(best_loss, avg_loss):.6f}"
        }])

        # 判断并保存“全场最佳”模型，供测试脚本 baseline_swin_us.py 读取
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), best_path)
            log_msg = f"⭐ Epoch {epoch}: Swin-US 达到新的性能峰值 (Loss: {avg_loss:.6f})"
        else:
            log_msg = f"Epoch {epoch}: Loss = {avg_loss:.6f} | LR = {current_lr:.2e}"

        print(log_msg)
        logger.record_log(log_msg)

        # 定期存档，防止训练意外中断
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_epoch_{epoch}.pth"))

    logger.record_log(f"✨ Swin-US 训练圆满完成。最优损失: {best_loss:.6f}")
    print(f"✅ 实验数据已存至: {RESULTS_DIR}/{MODEL_NAME}")


if __name__ == "__main__":
    train()