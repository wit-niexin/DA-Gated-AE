"""
Module: Test Runner
Description:
    全量测试调度中心。支持：
    1. 传统基准与深度学习基准: models/baselines/baseline_xxx.py
    2. 提出的模型: models/evaluators/test_da_gated_ae.py
"""

import argparse
import sys
import os
import importlib

# --- 1. 路径标准化 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="PCOS Denoising Global Evaluator")
    parser.add_argument('--model', type=str, required=True,
                        choices=['dunet', 'us_drunet', 'swin_us', 'da_gated_ae', 'ibf', 'srad', 'bm3d_us'],
                        help="选择要评估的模型")
    args = parser.parse_args()

    # --- 2. 路由逻辑：根据模型名称确定脚本路径与方法名 ---
    if args.model == 'da_gated_ae':
        # 对应提出的模型逻辑
        module_path = "models.evaluators.test_da_gated_ae"
        func_name = "run_da_gated_ae_experiment"
    else:
        # 对应那 5 种基准算法逻辑
        module_path = f"models.baselines.baseline_{args.model}"
        func_name = f"run_{args.model}_experiment"

    # --- 3. 动态执行 ---
    try:
        print(f"🔍 正在从 {module_path} 载入评估任务...")
        module = importlib.import_module(module_path)

        if hasattr(module, func_name):
            eval_func = getattr(module, func_name)

            print(f"🚀 启动评估: {args.model.upper()}")
            print(f"📊 实验设置: Noise levels = [0.001, 0.02, 0.5]")

            eval_func()  # 执行该脚本定义的完整评估流程

        else:
            print(f"❌ 错误: 在 {module_path} 中找不到方法 '{func_name}'")

    except ImportError as e:
        print(f"❌ 错误: 无法导入 {module_path}。请检查文件是否存在。")
        print(f"详情: {e}")


if __name__ == "__main__":
    main()