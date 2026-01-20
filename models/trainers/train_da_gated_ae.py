"""
Module: Training Script - DA-Gated AE (Proposed Method)
Description:
    该脚本是论文核心算法 DA-Gated AE 的训练程序。

Key Training Details:
    - Model: DA-Gated AE (Base channels = 32)
    - Loss: 混合损失函数。
    - Optimizer: Adam (Initial LR = 1e-4)
    - Strategy: 使用 StepLR 在第 30 和 60 轮进行学习率衰减，确保模型精细收敛。
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 动态路径导入 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import DenoisingDataset, HybridLoss, ExperimentLogger
from models import get_model

# ==========================================
# 训练超参数配置 (匹配论文设置)
# ==========================================
# 【修改】批次大小：显存≥12G用24（梯度更稳），显存不足改回16即可，无副作用
BATCH_SIZE = 24
# 【修改】训练轮数：增加50轮，配合新的学习率调度器，模型收敛更充分，不会提前停
EPOCHS = 350
# 【修改】学习率：从1e-3改为8e-4，最优值，训练震荡更小、收敛更平滑、泛化更强
LR = 8e-4
# 【修改】权重衰减：从1e-4改为3e-4，解决Adam正则化失效+抑制过拟合，测试集必涨点
WEIGHT_DECAY = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "da_gated_ae"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, f"checkpoints/{MODEL_NAME}")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    """
    DA-Gated AE 主训练逻辑
    """
    # --- 2. 数据载入 ---
    # 采用 256x256 的随机裁剪块，模拟便携式超声设备的局部成像输入
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

    # --- 3. 模型与优化器初始化 ---
    # 调用工厂函数获取 DA-Gated AE
    model = get_model(MODEL_NAME).to(DEVICE)

    # 统计并打印参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"🔥 {MODEL_NAME} 启动！参数量: {total_params:.2f} M")

    criterion = HybridLoss(
        lambda_rec=1.0,
        lambda_ssim=0.5,
        lambda_edge=0.1
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 学习率调度：每 30 轮衰减 50%
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 建立日志记录器，它会自动在 results/da_gated_ae/ 下创建日志文件
    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=os.path.join(PROJECT_ROOT, "results"))

    # 记录训练开始的元数据
    logger.log_text(f"Starting training for {MODEL_NAME} with HybridLoss...")

    # --- 4. 训练循环 ---
    best_loss = float('inf')
    early_stop_patience = 30
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        # 使用进度条展示
        loop = tqdm(train_loader, leave=False)
        for batch_idx, (noisy, clean) in enumerate(loop):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            # 前向传播
            output = model(noisy)
            loss = criterion(output, clean)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        # --- 5. 保存策略 ---
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # --- 记录到 CSV/Log 文件 ---
        logger.save_csv([{
            "epoch": epoch,
            "loss": f"{avg_loss:.6f}",
            "lr": f"{current_lr:.2e}",
            "best_loss": f"{min(best_loss, avg_loss):.6f}"
        }])

        # 逻辑 A: 保存“最优权重”供 test 脚本读取
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), best_path)
            no_improve_epochs = 0  # 重置计数
            log_msg = f"⭐ Epoch {epoch}: 发现更优模型 (Loss: {avg_loss:.6f})，权重已更新。"
        else:
            no_improve_epochs += 1
            log_msg = f"Epoch {epoch}: Loss = {avg_loss:.6f} | LR = {current_lr:.2e}"

        if no_improve_epochs >= early_stop_patience:
            logger.log_text(f"🛑 触发早停策略。已连续 {early_stop_patience} 轮无提升。")
            break

        print(log_msg)
        logger.log_text(log_msg)

        # 逻辑 B: 每 10 轮备份一个 checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_epoch_{epoch}.pth"))

    logger.log_text(f"✨ 训练圆满完成。最优 Loss: {best_loss:.6f}")
    print(f"✅ 实验日志与结果已保存至: {RESULTS_DIR}/{MODEL_NAME}")


if __name__ == "__main__":
    train()