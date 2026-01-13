"""
脚本功能：PCOS超声图像数据集标准化预处理
实现目标：
    1. 格式转换：将原始RGB图像转换为单通道灰度图。
    2. 尺寸对齐：统一调整分辨率为 256x256。
    3. 数据划分：按照 8:1:1 的比例严格划分训练集、验证集和测试集。
    4. 噪声建模：根据论文公式实现乘性散斑噪声（Speckle Noise）的离线生成。
"""

import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from utils import SpeckleNoiseFactory


def add_speckle_noise(image_array, sigma_sq):
    """
    【核心数学实现：乘性散斑噪声模型】
    根据论文公式: Y = X + X * N
    其中:
        X 是干净图像 (Ground Truth)
        N 是均值为 0, 方差为 sigma_sq 的高斯噪声
        Y 是生成的含噪图像 (Noisy Image)

    参数:
        image_array: 原始灰度图像矩阵 [0, 255]
        sigma_sq: 噪声方差 (论文中定义的等级：0.001, 0.02, 0.5)
    """
    # 步骤 1: 归一化处理
    # 神经网络训练通常在 [0, 1] 空间进行，此处先转为浮点数，避免 uint8 溢出
    image_f = image_array.astype(np.float32) / 255.0

    # 步骤 2: 生成高斯分布的噪声矩阵 N
    # np.random.normal 参数为 (均值, 标准差, 输出形状)
    noise = np.random.normal(0, np.sqrt(sigma_sq), image_f.shape)

    # 步骤 3: 应用乘性模型 (Multiplicative Model)
    # 不同于常见的加性噪声 (X + N)，散斑噪声随信号强度增大而增大，这是超声成像的物理特性
    noisy_f = image_f + image_f * noise

    # 步骤 4: 后处理
    # 裁剪 (Clip) 保证像素值不超出 [0, 1] 范围，然后恢复到 [0, 255] 的 8-bit 格式
    noisy_f = np.clip(noisy_f, 0, 1)
    return np.uint8(noisy_f * 255)


def prepare_data():
    """
    数据集准备主函数：负责路径管理、数据划分与图像处理流水线
    """
    # --- 路径与配置设定 ---
    src_dir = "./data/raw"  # 原始采集的 PCOS 图像存放处
    target_base = "./data"  # 处理后的标准化数据集根目录
    target_size = (256, 256)  # 统一的分辨率
    sigmas = [0.001, 0.02, 0.5]  # 对应低、中、高三种噪声水平
    modes = ['train', 'val', 'test']  # 文件夹分配

    # --- 步骤 1: 随机划分数据集 ---
    # 获取所有图片文件名
    all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.seed(42)  # 固定随机种子，确保实验可重复性
    random.shuffle(all_files)

    num_imgs = len(all_files)
    train_end = int(num_imgs * 0.8)  # 80% 用于训练
    val_end = train_end + int(num_imgs * 0.1)  # 10% 用于验证

    # 建立映射字典，将文件名分配给不同的用途
    splits = {
        'train': all_files[:train_end],
        'val': all_files[train_end:val_end],
        'test': all_files[val_end:]
    }

    # --- 步骤 2: 核心处理循环 ---
    for mode in modes:
        print(f"--- 正在处理 {mode} 集 (共 {len(splits[mode])} 张) ---")

        # A. 创建干净图存放路径
        clean_dir = os.path.join(target_base, mode, "clean")
        os.makedirs(clean_dir, exist_ok=True)

        # B. 创建不同噪声等级的存放路径
        # 此处预先生成含噪图片并保存在磁盘，会占用更多空间，
        # 但训练时加载速度快，且能确保每次 Epoch 模型看到的噪声完全一致。
        noisy_dirs = {}
        for s in sigmas:
            d = os.path.join(target_base, mode, f"noisy_{s}")
            os.makedirs(d, exist_ok=True)
            noisy_dirs[s] = d

        for filename in tqdm(splits[mode]):
            # 1. 读取原始图像
            img_path = os.path.join(src_dir, filename)
            img_rgb = cv2.imread(img_path)
            if img_rgb is None:
                continue

            # 2. 灰度化处理
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            # 3. 尺寸变换
            # 使用 INTER_CUBIC (双三次插值)，相比线性插值能更好地保留超声图像中的细小卵泡边缘
            img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_CUBIC)

            # 4. 保存干净图
            cv2.imwrite(os.path.join(clean_dir, filename), img_resized)

            # 5. 生成并保存噪声图
            # 论文中提到在训练阶段 (Training Phase) 噪声是“Online”生成的。
            # 意味着在训练脚本的 DataLoader 中实时调用 add_speckle_noise。
            # 本脚本目前采用 "Offline" 方式，为测试集 (Test Set) 提供固定的评价标准。
            for s in sigmas:
                # 这里可以调用本地实现的噪声方法，也可以调用噪声工厂的方法。
                # img_noisy = add_speckle_noise(img_resized, s)
                img_noisy = SpeckleNoiseFactory.add_speckle_noise(img_resized, s)
                cv2.imwrite(os.path.join(noisy_dirs[s], filename), img_noisy)


if __name__ == "__main__":
    # 执行前请确保 ./data/raw 文件夹下已放入原始 PCOS 图像
    prepare_data()
    print("\n✅ 数据集准备完毕！目录结构已按 [clean/noisy_sigma] 自动生成。")
