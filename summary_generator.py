"""
Script: Summary Generator for Information Sciences
Description:
    自动化实验数据汇总脚本。
    1. 读取 results/csv_data/ 下所有算法的 CSV 结果。
    2. 将数据合并为一个总表。
    3. 自动识别最优指标（如最高 PSNR，最低 Params），并在 LaTeX 输出中自动加粗。
    4. 导出为 .csv (供 Excel 查看) 和 .tex (供论文排版)。
"""

import os
import pandas as pd
import numpy as np
import re

# ==========================================
# 配置区域
# ==========================================
RESULTS_DIR = "results/csv_data"
OUTPUT_FILENAME = "final_experiment_summary"

# 定义算法显示的顺序
ORDERED_MODELS = [
    'ibf', 'srad', 'bm3d_us',  # Traditional
    'dunet', 'us_drunet', 'swin_us',  # DL Baselines
    'da_gated_ae'  # Ours
]

# 定义指标的方向 (True=越大越好, False=越小越好)
METRIC_DIRECTIONS = {
    'PSNR': True,
    'SSIM': True,
    'RMSE': False,
    'ENL': True,
    'CNR': True,
    'EPI': True,
    'Params(M)': False,
    'GFLOPs': False,
    'FPS': True
}


def parse_mean_value(val_str):
    """
    从 "28.45 ± 0.12" 格式的字符串中提取均值 28.45。
    如果是 "-" 或非数值，返回 None。
    """
    if not isinstance(val_str, str):
        return val_str
    if "±" in val_str:
        try:
            return float(val_str.split('±')[0].strip())
        except:
            return None
    try:
        return float(val_str)
    except:
        return None


def generate_latex_code(df):
    """
    生成符合 Information Sciences 标准的 LaTeX 三线表代码。
    自动对每一列的最优值加粗 (\textbf{})。
    """
    # 复制一份用于处理
    latex_df = df.copy()

    # 获取唯一的噪声等级
    noise_levels = df['Noise'].unique()

    # 针对每个噪声等级，进行组内比较并加粗
    for noise in noise_levels:
        # 获取当前噪声等级的索引 mask
        mask = latex_df['Noise'] == noise

        # 遍历需要比较的列
        for col, is_higher_better in METRIC_DIRECTIONS.items():
            if col not in latex_df.columns:
                continue

            # 提取数值列表
            raw_values = latex_df.loc[mask, col].apply(parse_mean_value)

            # 过滤掉 None (比如传统算法的 Params 为 "-")
            valid_indices = raw_values.dropna().index
            if valid_indices.empty:
                continue

            # 找到最优值的索引
            if is_higher_better:
                best_idx = raw_values.idxmax()
            else:
                best_idx = raw_values.idxmin()

            # 给最优值添加 LaTeX 加粗标记
            original_str = latex_df.loc[best_idx, col]
            latex_df.loc[best_idx, col] = f"\\textbf{{{original_str}}}"

    # 开始构建 LaTeX 字符串
    latex_code = []
    latex_code.append("\\begin{table*}[htbp]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{Quantitative comparison on ultrasound datasets. Best results are \\textbf{bold}.}")
    latex_code.append("\\resizebox{\\textwidth}{!}{")
    # 自动生成列格式: l c c ...
    col_format = "l" + "c" * (len(df.columns) - 1)
    latex_code.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_code.append("\\toprule")

    # 表头
    headers = " & ".join(df.columns)
    latex_code.append(f"{headers} \\\\")
    latex_code.append("\\midrule")

    # 内容行
    current_noise = None
    for _, row in latex_df.iterrows():
        # 为了美观，如果是新的噪声等级，可以加一个空行或分割线 (此处简化，直接打印)
        row_str = " & ".join([str(x) for x in row.values])
        latex_code.append(f"{row_str} \\\\")

    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}}")
    latex_code.append("\\label{tab:main_results}")
    latex_code.append("\\end{table*}")

    return "\n".join(latex_code)


def main():
    print(f"🚀 开始汇总实验数据...")

    all_data = []

    # 1. 按照预定顺序读取 CSV
    for model in ORDERED_MODELS:
        # 尝试匹配文件名 (兼容 baseline_ibf_results.csv 这种格式)
        csv_name = f"baseline_{model}_results.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_name)

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # 确保 Method 列名称统一
                if 'Method' not in df.columns:
                    df.insert(0, 'Method', model.upper())
                all_data.append(df)
                print(f"  ✅ 已加载: {model}")
            except Exception as e:
                print(f"  ❌ 加载失败 {model}: {e}")
        else:
            print(f"  ⚠️ 未找到文件: {csv_path} (跳过)")

    if not all_data:
        print("没有找到任何数据文件！请先运行测试脚本。")
        return

    # 2. 合并数据
    final_df = pd.concat(all_data, ignore_index=True)

    # 3. 数据清洗与排序
    # 按照 Noise (升序) -> Method (保持列表顺序) 排序
    # 为了保持自定义顺序，将 Method 转为 Categorical
    final_df['Method'] = pd.Categorical(
        final_df['Method'],
        categories=[m for m in final_df['Method'].unique()],
        ordered=True
    )
    final_df = final_df.sort_values(by=['Noise', 'Method'])

    # 4. 导出 CSV
    csv_output_path = f"{OUTPUT_FILENAME}.csv"
    final_df.to_csv(csv_output_path, index=False)
    print(f"\n📊 汇总 CSV 已生成: {csv_output_path}")

    # 5. 导出 LaTeX 代码
    latex_source = generate_latex_code(final_df)
    tex_output_path = f"{OUTPUT_FILENAME}.tex"
    with open(tex_output_path, "w", encoding="utf-8") as f:
        f.write(latex_source)

    print(f"📝 LaTeX 源码已生成: {tex_output_path}")
    print("-" * 30)
    print("提示: 你可以直接打开 .tex 文件，复制内容到论文中。")


if __name__ == "__main__":
    main()
