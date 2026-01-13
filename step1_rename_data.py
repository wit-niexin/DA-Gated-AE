# 这是数据集的准备工作，将患者组和健康组的图像分别编号并存放进data/raw/文件夹中

import os
import shutil


def rename_and_move(src_path, prefix, start_count, save_dir):
    """
    src_path: 原始图片的文件夹路径
    prefix: 重命名的前缀 (img_infected 或 img_notinfected)
    start_count: 起始编号
    save_dir: 目标 raw 文件夹路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取目录下所有图片文件
    files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()  # 排序确保编号有一定顺序

    print(f"开始处理 {prefix}，共 {len(files)} 张图片...")

    count = start_count
    for filename in files:
        # 构造新文件名，例如: img_infected_0001.jpg
        # :04d 表示编号占4位，不足补0
        extension = os.path.splitext(filename)[1]
        new_name = f"{prefix}_{count: 04d}{extension}"

        src_file = os.path.join(src_path, filename)
        dst_file = os.path.join(save_dir, new_name)

        # 移动并重命名
        shutil.copy(src_file, dst_file)
        count += 1

    print(f"{prefix} 处理完成，当前编号至: {count - 1}")


# --- 主程序执行 ---
if __name__ == "__main__":
    # 1. 修改这里的路径为电脑上的实际路径
    INFECTED_DIR = "C:/Users/Administrator/data/test/infected"
    NOT_INFECTED_DIR = "C:/Users/Administrator/data/test/notinfected"

    # 2. 项目 raw 目录
    RAW_TARGET_DIR = "./data/raw"

    # 执行重命名
    # 患者组：img_infected_0001.jpg ...
    rename_and_move(INFECTED_DIR, "img_infected", 1, RAW_TARGET_DIR)

    # 非患者组：img_notinfected_0001.jpg ...
    rename_and_move(NOT_INFECTED_DIR, "img_notinfected", 1, RAW_TARGET_DIR)

    print(f"\n所有图片已成功存入: {os.path.abspath(RAW_TARGET_DIR)}")
    print(f"总计图片张数: {len(os.listdir(RAW_TARGET_DIR))}")
