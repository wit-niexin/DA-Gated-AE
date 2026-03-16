"""
Module: Training Script - DA-Gated AE (Proposed Method)
Description:
    该脚本实现了 DA-Gated AE 的静态数据集训练流程，以确保与 Baseline 模型在相同噪声样本下进行公平对比。
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
# --- 手动配置消融实验参数 ---
ablation_config = {
    "use_gate": True,  # Model A 改为 False，Model B/C/D 改为 True
    "use_dsc": True    # 保持 True
}
# --- 手动配置损失函数参数 ---
l_rec, l_ssim, l_edge = 1.0, 0.5, 0.1 # Model D 权重

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, f"checkpoints/{MODEL_NAME}")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def validate(model, val_loader):
    """验证集评估逻辑：计算HybridLoss（与训练目标一致）"""
    model.eval()
    total_loss = 0
    count = 0
    criterion = HybridLoss(lambda_rec=l_rec, lambda_ssim=l_ssim, lambda_edge=l_edge).to(DEVICE)
    
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item()
            count += 1
    
    return total_loss / count


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
    model = get_model(MODEL_NAME, **ablation_config).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = HybridLoss(lambda_rec=l_rec, lambda_ssim=l_ssim, lambda_edge=l_edge).to(DEVICE)
    logger = ExperimentLogger(model_name=MODEL_NAME, root_dir=RESULTS_DIR)

    best_loss = float('inf')  # 改为基于loss保存（越小越好）
    patience_count = 0
    EARLY_STOP_PATIENCE = 30

    print(f"🔥 {MODEL_NAME} 训练启动 (HybridLoss保存策略)")
    print("📊 保存策略: 基于HybridLoss保存最优模型（与训练目标一致）")

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
            val_loss = validate(model, val_loader)
            print(f"Epoch {epoch}: Loss={avg_loss:.5f} | Val Loss={val_loss:.5f}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth"))
                print(f"⭐ 发现新低！权重已保存至 {MODEL_NAME}_best.pth")
            else:
                patience_count += 1

            if patience_count >= EARLY_STOP_PATIENCE:
                print("🛑 触发早停。")
                break

        # 记录日志
        logger.save_csv([{"epoch": epoch, "loss": avg_loss, "best_loss": best_loss}])

    print(f"✅ 训练圆满完成。最低 Loss: {best_loss:.5f}")


if __name__ == "__main__":
    train()
