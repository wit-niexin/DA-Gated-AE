"""
Description:
    实现论文描述的 Swin-US。
    特点：
    - 使用 DSC 进行 Patch Embedding。
    - 基于 Swin Block 的 Encoder/Decoder。
    - 双域通道-空间注意力机制。
"""
import torch
import torch.nn as nn


# --- 基础组件：深度可分离卷积 (DSC) ---
class DSCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class WindowAttention(nn.Module):
    """
    简化版 Swin Window Attention。
    将图像切分为 window_size x window_size 的窗口，在窗口内计算 Attention。
    """

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        ws = self.window_size

        # 0. Shift Window Logic
        shift_size = ws // 2
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        else:
            shifted_x = x
            
        # 1. Padding if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        # 2. Window Partition [B, C, H, W] -> [B * num_windows, ws*ws, C]
        # 变换为: [B, C, Hp//ws, ws, Wp//ws, ws] -> [B, Hp//ws, Wp//ws, ws, ws, C]
        x_windows = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)

        # 3. Attention
        x_windows = self.norm1(x_windows)
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        x_windows = x_windows + attn_out
        x_windows = x_windows + self.mlp(self.norm2(x_windows))

        # 4. Window Reverse
        x = x_windows.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)

        # 5. Reverse Shift
        if shift_size > 0:
            x = torch.roll(x_windows, shifts=(shift_size, shift_size), dims=(2, 3))
        else:
            x = x_windows
            
        return x


class SwinUS(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, embed_dim=48, window_size=8):
        super(SwinUS, self).__init__()
        self.patch_embed = DSCBlock(in_channels, embed_dim)

        # 使用 Window Attention
        self.enc1 = WindowAttention(embed_dim, window_size, num_heads=3)
        self.enc2 = WindowAttention(embed_dim * 2, window_size, num_heads=6)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.bottleneck = WindowAttention(embed_dim * 4, window_size, num_heads=12)

        # 简单的线性投影用于调整通道
        self.proj_e2 = nn.Conv2d(embed_dim, embed_dim * 2, 1)
        self.proj_b = nn.Conv2d(embed_dim * 2, embed_dim * 4, 1)

        self.dec2 = nn.Conv2d(embed_dim * 6, embed_dim * 2, 1)
        self.dec1 = nn.Conv2d(embed_dim * 3, embed_dim, 1)
        self.final = nn.Conv2d(embed_dim, out_channels, 1)

    def forward(self, x):
        x = self.patch_embed(x)
        e1 = self.enc1(x)  # [B, 48, H, W]

        e2_in = self.down(e1)
        e2_in = self.proj_e2(e2_in)  # [B, 96, H/2, W/2]
        e2 = self.enc2(e2_in)

        b_in = self.down(e2)
        b_in = self.proj_b(b_in)  # [B, 192, H/4, W/4]
        b = self.bottleneck(b_in)

        # Decoder
        d2 = self.up(b)
        # Direct Concatenation
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
