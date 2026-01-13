"""
Module: Model Factory
Description:
    模型工厂模块。负责统一管理所有网络架构。
    支持：DU-Net (Dense U-Net), Swin-US (Transformer), DA-Gated AE (Proposed).
"""

import torch
from .networks.dunet import DUNet
from .networks.swin_us import SwinUS
from .networks.da_gated_ae import DAGatedAE


def get_model(model_name: str, **kwargs):
    """
    模型工厂函数：根据模型名称返回模型实例。

    Args:
        model_name (str): 模型的标识符 (如 'dunet', 'swin_us', 'da_gated_ae')。
        **kwargs: 传递给模型构造函数的参数。
                 常用参数: in_channels=1, out_channels=1, base_feat=32 (具体视模型而定)

    Returns:
        torch.nn.Module: 实例化的模型对象。
    """

    models_map = {
        "dunet": DUNet,
        "swin_us": SwinUS,
        "da_gated_ae": DAGatedAE,
    }

    name_lower = model_name.lower()

    if name_lower not in models_map:
        supported = ", ".join([f"'{m}'" for m in models_map.keys()])
        raise ValueError(
            f"\n❌ [ModelFactory Error]: 找不到模型 '{model_name}'。"
            f"\n✅ 当前支持的模型列表为: [{supported}]"
            f"\n💡 请检查 models/networks/ 目录下是否有对应的实现，并在 models/__init__.py 中注册。"
        )

    # 针对超声图像去噪任务，默认输入输出均为 1 通道
    # 如果调用时没传，则补全；如果传了，则以传的为准
    if 'in_channels' not in kwargs:
        kwargs['in_channels'] = 1
    if 'out_channels' not in kwargs:
        kwargs['out_channels'] = 1

    # 获取类并实例化
    model_class = models_map[name_lower]
    print(f"🛠️  Model Factory: 正在构建 {model_class.__name__} (params: {kwargs})")

    return model_class(**kwargs)


# --- 公开接口 ---
__all__ = [
    'get_model',
    'DUNet',
    'SwinUS',
    'DAGatedAE'
]
