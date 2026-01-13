"""
Module: Main Entry - Train Runner
Description:
    统一训练脚本。支持训练提出的 DA-Gated AE 以及所有深度学习基准模型。
"""

import argparse
import subprocess
import os
import sys


def main():
    # --- 1. 参数解析器配置 ---
    parser = argparse.ArgumentParser(description="PCOS Ultrasound Denoising Training Runner")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['dunet', 'swin_us', 'da_gated_ae'],
        help="选择要训练的模型: 'dunet' (CNN基准), 'swin_us' (Transformer基准), 'da_gated_ae' (本文提出)"
    )

    args = parser.parse_args()

    # --- 2. 脚本路径映射 ---
    scripts_map = {
        "dunet": "models/trainers/train_dunet.py",
        "swin_us": "models/trainers/train_swin_us.py",
        "da_gated_ae": "models/trainers/train_da_gated_ae.py"
    }

    script_path = scripts_map[args.model]

    # --- 3. 运行环境检查 ---
    if not os.path.exists(script_path):
        print(f"❌ 错误: 找不到训练脚本 '{script_path}'。请检查文件是否存在。")
        sys.exit(1)

    # --- 4. 启动训练流程 ---
    print(f"🚀 正在启动 {args.model.upper()} 训练任务...")

    try:
        # 使用 subprocess 调用具体的训练脚本
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"💥 训练意外中断: {e}")
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止了训练。")


if __name__ == "__main__":
    main()