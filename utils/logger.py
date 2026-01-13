"""
Module: ExperimentLogger
Description:
    实验日志与结果管理模块。负责自动创建实验结果目录、保存去噪后的图像、
    生成定性对比图（Qualitative Comparison）、并以 CSV 和 .log 格式持久化实验指标（PSNR, SSIM等）。
    该模块确保了不同模型（如 IBF, DUNet, Swin-US）的实验评估流程具有统一的输出格式。

Directory Structure:
    results/
    ├── csv_data/          -> 存储各模型的指标汇总 (CSV)
    ├── logs/              -> 存储所有实验的历史运行记录 (LOG)
    ├── denoised_images/   -> 存储完整的去噪后图像结果
    └── qualitative/       -> 存储用于论文展示的对比拼图 (Clean | Noisy | Denoised)
"""

import os
import time
import cv2
import pandas as pd
import numpy as np


class ExperimentLogger:
    """
    实验日志记录器类，用于自动化管理深度学习和传统算法的测试产出。
    """

    def __init__(self, model_name, root_dir="../../results"):
        """
        初始化日志记录器，并自动创建必要的文件系统架构。

        Args:
            model_name (str): 模型名称，用于区分不同的实验记录（如 'dunet'）。
            root_dir (str): 结果存储的根目录，默认为项目根目录下的 results。
        """
        self.model_name = model_name
        # 使用 abspath 确保在不同目录下运行脚本时路径逻辑一致
        self.results_dir = os.path.abspath(root_dir)
        self.logs_dir = os.path.join(self.results_dir, "logs")
        self.csv_dir = os.path.join(self.results_dir, "csv_data")

        # 确保目录存在
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def save_csv(self, results_list):
        """
        将实验汇总数据（列表形式）保存为 CSV 文件。

        Args:
            results_list (list): 包含各噪声等级指标的字典列表。
        Returns:
            str: 最终保存的 CSV 文件路径。
        """
        df = pd.DataFrame(results_list)
        csv_path = os.path.join(self.csv_dir, f"baseline_{self.model_name}_results.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def record_log(self, df_summary):
        """
        将本次实验的详细参数和平均指标追加到 experiment_history.log 中，并打印到控制台。

        Args:
            df_summary (DataFrame): Pandas DataFrame 格式的指标汇总表。
        Returns:
            str: 日志文件路径。
        """
        log_file = os.path.join(self.logs_dir, "experiment_history.log")
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 25} 实验记录 {'=' * 25}\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"算法/模型: {self.model_name.upper()}\n")
            f.write("-" * 60 + "\n")
            f.write(df_summary.to_string(index=False))
            f.write(f"\n{'=' * 60}\n")

        # 终端回显
        print("\n" + "=" * 85)
        print(f" 论文 Baseline 数据汇总: {self.model_name.upper()} ")
        print(df_summary.to_string(index=False))
        print("=" * 85)

        return log_file

    def save_images(self, name, sigma_val, clean, noisy, denoised, save_path, qualitative_dir, img_names_list):
        """
        执行图像持久化逻辑。

        1. 存储独立的去噪图像用于后续全量计算。
        2. 自动挑选前 N 张图片生成对比长图，直接用于论文排版展示。

        Args:
            name (str): 图像文件名。
            sigma_val (float): 对应的噪声强度水平。
            clean (ndarray): 真实清晰图像（Ground Truth）。
            noisy (ndarray): 输入的含噪声图像。
            denoised (ndarray): 模型输出的去噪图像。
            save_path (str): 去噪图保存目录。
            qualitative_dir (str): 对比拼图保存目录。
            img_names_list (list): 用于切片选择少量图片生成对比图。
        """
        # 1. 保存单张去噪后的结果
        cv2.imwrite(os.path.join(save_path, name), denoised)

        # 2. 生成定性展示图 (Qualitative Visualization)
        # 目前仅选择该等级下的前 2 张图片，以后再优化选择逻辑
        if name in img_names_list[:2]:
            # 拼接：左(原图) | 中(噪声图) | 右(去噪图)
            # np.hstack 要求所有图像尺寸必须一致
            comparison = np.hstack((clean, noisy, denoised))
            qual_path = os.path.join(qualitative_dir, f"compare_sigma_{sigma_val}_{name}")
            cv2.imwrite(qual_path, comparison)
