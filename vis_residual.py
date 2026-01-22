import cv2
import numpy as np


def generate_residual_map(img_gt, img_denoised, save_path=None, amplify_factor=1.0):
    """
    生成误差残差热力图 (Error Residual Map)

    Args:
        img_gt (numpy.ndarray): Ground Truth 图像 (灰度, 0-255)
        img_denoised (numpy.ndarray): 去噪后的图像 (灰度, 0-255)
        save_path (str): 保存路径 (可选)
        amplify_factor (float): 误差放大因子。某些时候误差很小，为了看清分布，论文中常会把差值乘以一个系数 (如 3 或 5) 来增强对比度。

    Returns:
        heatmap_img: 伪彩色残差图
    """
    # 1. 计算绝对差值
    # 注意：必须先转为 float 计算，防止 uint8 溢出
    diff = np.abs(img_gt.astype(np.float32) - img_denoised.astype(np.float32))

    # 2. (可选) 放大误差以便观察
    diff = diff * amplify_factor

    # 3. 截断并归一化到 0-255
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # 4. 应用 Jet Colormap (蓝-青-黄-红)
    # cv2.COLORMAP_JET 是最常用的，蓝色=低，红色=高
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    # 5. 保存或返回，使用这个方法时根据需求修改
    if save_path:
        cv2.imwrite(save_path, heatmap)
        print(f"✅ Residual Map saved to: {save_path}")

    return heatmap


# --- 使用示例 ---
if __name__ == "__main__":
    # 读入图片
    # gt = cv2.imread('clean.png', 0)
    # pred = cv2.imread('denoised_dunet.png', 0)

    # 生成对比图
    # generate_residual_map(gt, pred, "fig8_dunet_residual.png", amplify_factor=3.0)
    pass
