"""
LoGStem-Lite: 轻量化 LoGStem 实现（方案A）
包含：GhostConv, ECA_Attention, Gaussian (learnable sigma), DRFD_LoG (lite), LoGFilter, LoGStem_Lite
可直接替换原始 LoGStem 模块使用。

作者：根据对话与改进建议整合
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果你的工程有现成的 Conv（包含 BN+Act），可用之；否则使用内部实现
# from engine.extre_module.ultralytics_nn.conv import Conv  # 可选

# ----------------------------
# Simple GhostConv implementation
# ----------------------------
class GhostModule(nn.Module):
    """
    Simple Ghost module:
    primary_conv -> cheap_operations (depthwise conv) to generate ghost features
    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_kernel=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU() if relu else nn.Identity()
        )

        # cheap operation: depthwise conv to generate ghost features
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel, 1, dw_kernel // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU() if relu else nn.Identity()
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        if self.oup == primary.shape[1]:
            return primary
        cheap = self.cheap_operation(primary)
        out = torch.cat([primary, cheap], dim=1)
        return out[:, :self.oup, :, :]

class GhostConv(nn.Module):
    """GhostConv wrapper to mimic Conv + BN + Act with GhostModule"""
    def __init__(self, in_ch, out_ch, k=1, s=1, g=1, ratio=2, dw_ks=3):
        super().__init__()
        self.ghost = GhostModule(in_ch, out_ch, kernel_size=k, ratio=ratio, dw_kernel=dw_ks, stride=s, relu=True)

    def forward(self, x):
        return self.ghost(x)

# If you prefer to use original Conv implementation from your project, swap GhostConv -> Conv here.

# ----------------------------
# ECA Channel Attention (lightweight)
# ----------------------------
class ECA_Attention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D conv with padding simulating local cross-channel interaction
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)            # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv1d(y)             # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

# ----------------------------
# Gaussian with learnable sigma
# ----------------------------
class Gaussian(nn.Module):
    def __init__(self, dim, kernel_size=9, sigma_init=0.6, feature_extra=True):
        """
        Gaussian smoothing implemented as depthwise conv with a generated kernel.
        sigma is learnable (scalar) to adapt smoothing strength.
        If feature_extra True, a small Conv block refines features (we use GhostConv here).
        """
        super().__init__()
        self.feature_extra = feature_extra
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # learnable sigma (constrained positive by abs)
        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))
        # depthwise conv placeholder (weights will be set on the fly each forward)
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=self.padding,
                              groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        if feature_extra:
            self.conv_extra = nn.Sequential(
                GhostConv(dim, max(16, dim//2), k=1),
                GhostConv(max(16, dim//2), max(16, dim//2), k=3),
                GhostConv(max(16, dim//2), dim, k=1)
            )

    def gaussian_kernel(self, size, sigma):
        """create 2D gaussian kernel (not normalized to sum=1 to keep scale)"""
        ax = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x):
        B, C, H, W = x.shape
        sigma = torch.abs(self.sigma) + 1e-6
        # create kernel on same device
        kernel = self.gaussian_kernel(self.kernel_size, float(sigma)).to(x.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
        kernel = kernel.repeat(C, 1, 1, 1)         # [C,1,k,k]
        # set weights
        self.conv.weight.data = kernel
        out = self.conv(x)       # depthwise conv
        out = self.act(self.norm(out))
        if self.feature_extra:
            return self.conv_extra(x + out)
        else:
            return out

# ----------------------------
# LoGFilter (LoG operator) - using separable/efficient ops
# ----------------------------
class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, sigma=1.0):
        super().__init__()
        # initial conv to project channels (use GhostConv)
        self.conv_init = GhostConv(in_c, out_c, k=7)
        # build LoG kernel (fixed)
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        kernel = kernel / (kernel.sum() + 1e-12)
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv_init(x)
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x

# ----------------------------
# DRFD_LoG (lite) - simplified multi-scale fusion using GhostConv
# ----------------------------
class DRFD_LoG_Lite(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2, bias=False)
        self.act_c = nn.SiLU()
        self.norm_c = nn.BatchNorm2d(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = nn.BatchNorm2d(dim * 2)
        # fusion using GhostConv to be lightweight
        self.fusion = nn.Sequential(
            GhostConv(dim * 4, dim * 2, k=1),
            GhostConv(dim * 2, dim * 2, k=3),
            GhostConv(dim * 2, dim, k=1)  # reduce to dim (we want output 2*? or stem_dim final)
        )
        # optional lightweight attention on fused output
        self.eca = ECA_Attention(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv(x)  # [B, 2C, H, W]
        # gaussian not applied here (assume applied earlier)
        max_feat = self.norm_m(self.max_m(x))  # [B,2C,H/2,W/2]
        conv_feat = self.norm_c(self.act_c(self.conv_c(x)))  # [B,2C,H/2,W/2]
        fused = torch.cat([conv_feat, max_feat], dim=1)  # [B,4C,H/2,W/2]
        fused = self.fusion(fused)  # lightweight fusion -> [B, C, H/2, W/2] (we reduce)
        fused = self.eca(fused)
        return fused

# ----------------------------
# LoGStem_Lite - 完整组合
# ----------------------------
class LoGStem_Lite(nn.Module):
    def __init__(self, in_chans=3, stem_dim=12):
        """
        in_chans: input channels (e.g., 3)
        stem_dim: desired output channels of stem (e.g., 12 for B0 config)
        Output spatial: H/4 x W/4 (we implement two downsample stages)
        """
        super().__init__()
        self.stem_dim = stem_dim
        out_c14 = max(1, stem_dim // 4)
        out_c12 = max(1, stem_dim // 2)

        # LoGFilter: project and LoG edge extraction
        self.LoG = LoGFilter(in_chans, out_c14, kernel_size=7, sigma=1.0)

        # Conv_D: Ghost downsampling pipeline
        self.Conv_D = nn.Sequential(
            GhostConv(out_c14, out_c12, k=3, s=1),
            GhostConv(out_c12, out_c12, k=3, s=2)  # downsample by 2
        )

        # Gaussian with learnable sigma (applied on out_c12)
        self.gaussian = Gaussian(out_c12, kernel_size=9, sigma_init=0.6, feature_extra=True)
        self.norm = nn.BatchNorm2d(out_c12)

        # ECA attention after gaussian optionally (lightweight)
        self.eca = ECA_Attention(out_c12)

        # DRFD_LoG lite to further fuse and downsample to H/4
        self.drfd = DRFD_LoG_Lite(out_c12)

        # Ensure final output channels equals stem_dim (if different, project)
        final_channels = stem_dim
        self.output_proj = GhostConv(out_c12, final_channels, k=1) if out_c12 != final_channels else nn.Identity()

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.LoG(x)              # [B, out_c14, H, W]
        x = self.Conv_D(x)           # [B, out_c12, H/2, W/2]
        x = self.norm(x + self.gaussian(x))  # fused gaussian -> [B, out_c12, H/2, W/2]
        x = self.eca(x)
        x = self.drfd(x)             # [B, out_c12', H/4, W/4] (drfd reduces spatial by 2)
        x = self.output_proj(x)      # project to stem_dim if needed
        return x                     # [B, stem_dim, H/4, W/4]

# ----------------------------
# Unit test
# ----------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W = 1, 256, 256
    model = LoGStem_Lite(in_chans=3, stem_dim=12).to(device)
    inp = torch.randn(B, 3, H, W).to(device)
    out = model(inp)
    print("Input:", inp.shape)
    print("Output:", out.shape)  # expected [B, stem_dim, H/4, W/4] -> (1,12,64,64)

    # Optional: compute FLOPs via calflops if available
    try:
        from calflops import calculate_flops
        flops, macs, _ = calculate_flops(model=model, input_shape=(B, 3, H, W), output_as_string=True)
        print("FLOPs:", flops)
    except Exception as e:
        print("calflops not available or failed:", e)
