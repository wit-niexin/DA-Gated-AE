"""
Module: DenoisingDataset
Description:
    通用去噪数据加载类。支持“盲去噪”训练模式，即从多个噪声等级目录中随机采样图片，
    以增强模型的泛化能力。包含随机裁剪、翻转等数据增强操作。

Input:
    - Clean images: data/train/clean/
    - Noisy images: data/train/noisy_*/

Output:
    - (noisy_tensor, clean_tensor) in range [0, 1]
"""

import os
import random
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from utils import SpeckleNoiseFactory


class DenoisingDataset(Dataset):
    """
    通用去噪数据集类，支持：
    1. 自动从多个噪声等级文件夹中随机采样（盲去噪训练）
    2. 基础数据增强（随机翻转）
    3. 随机裁剪（Patch Training，可选）
    """

    def __init__(self, clean_dir, noisy_root, patch_size=None):
        """
        :param clean_dir: 干净图像的绝对路径 (如 data/train/clean)
        :param noisy_root: 噪声文件夹所在的根目录 (如 data/train)
        :param patch_size: 裁剪尺寸，若为 None 则返回整张原图
        """
        self.clean_dir = clean_dir
        self.patch_size = patch_size

        # 获取所有合法图片名
        self.img_names = [f for f in os.listdir(clean_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 自动识别 noisy_ 开头的子目录
        self.noisy_dirs = [
            os.path.join(noisy_root, d)
            for d in os.listdir(noisy_root)
            if os.path.isdir(os.path.join(noisy_root, d)) and d.startswith('noisy_')
        ]

        if not self.noisy_dirs:
            raise FileNotFoundError(f"在 {noisy_root} 中未找到任何以 'noisy_' 开头的文件夹！")

    def __len__(self):
        return len(self.img_names)

    # 这里用的是离线模式，即训练时读取已经生成的固定噪声图像
    def __getitem__(self, idx):
        name = self.img_names[idx]

        # 1. 加载图像 (0代表灰度模式)
        clean_path = os.path.join(self.clean_dir, name)
        clean_img = cv2.imread(clean_path, 0)

        # 随机选一个噪声强度文件夹，读取同名图片
        chosen_noisy_dir = random.choice(self.noisy_dirs)
        noisy_path = os.path.join(chosen_noisy_dir, name)
        noisy_img = cv2.imread(noisy_path, 0)

        if clean_img is None or noisy_img is None:
            # 这种异常处理在服务器大规模训练时很重要
            return self.__getitem__(random.randint(0, len(self.img_names) - 1))

        # 2. 转换为 Tensor (to_tensor 会自动将 0-255 映射到 0.0-1.0)
        clean_tensor = TF.to_tensor(clean_img)
        noisy_tensor = TF.to_tensor(noisy_img)

        # 3. 数据增强：随机裁剪 (Patch-based Training)
        # 这对于超声大图非常有用，可以节省显存并增加训练样本的多样性
        if self.patch_size:
            c, h, w = clean_tensor.shape
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                clean_tensor = TF.crop(clean_tensor, top, left, self.patch_size, self.patch_size)
                noisy_tensor = TF.crop(noisy_tensor, top, left, self.patch_size, self.patch_size)

        # 4. 数据增强：随机水平翻转
        if random.random() > 0.5:
            clean_tensor = TF.hflip(clean_tensor)
            noisy_tensor = TF.hflip(noisy_tensor)

        # 5. 数据增强：随机垂直翻转
        if random.random() > 0.5:
            clean_tensor = TF.vflip(clean_tensor)
            noisy_tensor = TF.vflip(noisy_tensor)

        return noisy_tensor, clean_tensor

    """
    # 下面的代码是在线生成噪声的模式
    def __getitem__(self, index):
        # 1. 加载干净的 Ground Truth (X)
        clean_img_tensor = self.load_clean_image(index)

        # 2. 从预设的噪声等级中随机抽取一个
        target_sigma = random.choice([0.001, 0.02, 0.5])

        # 3. 实时生成含噪图像 (Y)
        noisy_img_tensor = SpeckleNoiseFactory.add_speckle_noise(clean_img_tensor, target_sigma)

        return noisy_img_tensor, clean_img_tensor
    """