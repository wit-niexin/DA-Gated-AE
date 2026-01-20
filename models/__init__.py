"""
Module: Model Factory (Unified Architecture Management)
Description:
    模型工厂模块。作为本项目所有深度学习模型的统一入口。
    通过工厂模式（Factory Pattern），解耦了训练脚本与具体模型定义的依赖。

Supported Architectures:
    1. DA-Gated AE (Proposed): 基于深度可分离卷积与双重注意力门控的轻量化模型。
    2. US-DRUNet: 基于 Nakagami 统计先验嵌入的深层残差网络 (对标统计驱动方法)。
    3. DU-Net: 基于密集连接 (DenseBlock) 的改进型 U-Net (对标经典 CNN 方法)。
    4. Swin-US: 基于 Swin Transformer 的架构 (对标注意力机制方法)。

Design Principles:
    - 统一接口: 所有模型均通过 get_model 函数实例化。
    - 容错处理: 自动补充默认参数 (如 in/out channels)。
"""

import torch
from .networks.dunet import DUNet
from .networks.swin_us import SwinUS
from .networks.us_drunet import USDRUNet
from .networks.da_gated_ae import DAGatedAE


def get_model(model_name: str, **kwargs):
    """
    模型工厂核心函数。

    Args:
        model_name (str): 模型唯一标识符。可选: ['da_gated_ae', 'us_drunet', 'dunet', 'swin_us']
        **kwargs: 灵活的模型参数。
                 - in_channels: 输入通道 (默认 1)
                 - out_channels: 输出通道 (默认 1)
                 - nak_channels: 仅用于 us_drunet (默认 2: mu, omega)
                 - base_features: 基础特征图数量

    Returns:
        torch.nn.Module: 实例化的 PyTorch 模型对象。
    """

    # 1. 注册表映射 (将字符串标识符映射到具体的类对象)
    models_map = {
        "da_gated_ae": DAGatedAE,
        "us_drunet": USDRUNet,
        "dunet": DUNet,
        "swin_us": SwinUS,
    }

    name_lower = model_name.lower()

    # 2. 存在性检查与报错提示
    if name_lower not in models_map:
        error_msg = (
            f"\n❌ [ModelFactory Error]: 识别不到模型名称 '{model_name}'。"
            f"\n✅ 论文实验支持的深度学习模型包括: {list(models_map.keys())}"
        )
        print(error_msg)
        raise ValueError(f"Model {model_name} not supported.")

    # 3. 默认参数补全 (针对医学超声灰度图去噪任务)
    # 强制确保医学影像处理的默认通道为 1 (灰度图)
    defaults = {
        'in_channels': 1,
        'out_channels': 1,
    }

    # 针对不同模型的特性，注入特定默认超参数
    if name_lower == "us_drunet":
        # US-DRUNet 必须包含统计参数通道 (Nakagami mu, omega)
        defaults['nak_channels'] = 2
        defaults['base_features'] = 64
    elif name_lower == "da_gated_ae":
        # 我们的模型使用轻量化配置
        defaults['base_feat'] = 32

    final_params = {**defaults, **kwargs}

    # 4. 实例化模型
    model_class = models_map[name_lower]
    instance = model_class(**final_params)

    # 5. 打印构建信息
    # 计算模型总参数量 (单位: M, 百万)
    total_params = sum(p.numel() for p in instance.parameters()) / 1e6

    print(f"--- [Model Factory] 构建信息 ---")
    print(f"🚀 选定架构: {model_class.__name__}")
    print(f"📦 参数配置: {final_params}")
    print(f"📊 总参数量: {total_params: .2f} M")
    print(f"--------------------------------")

    return instance


# --- 公开接口暴露 ---
# 定义 __all__ 确保 from models import * 时只导出 get_model 及相关类
__all__ = [
    'get_model',
    'DUNet',
    'SwinUS',
    'USDRUNet',
    'DAGatedAE'
]
