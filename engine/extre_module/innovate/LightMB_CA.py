"""
本文件由BiliBili：魔傀面具整理
改进方案A（轻量化 + 通道注意力）
路径示例：engine/extre_module/module_images/TPAMI2025-MANet-A.png
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from calflops import calculate_flops

# ----------------------------
# 基础模块：Conv + DepthwiseConv（支持stride参数）
# ----------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    """自动计算padding，确保卷积后尺寸正确（尤其是stride>1时）"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class DWConv(nn.Module):
    """深度可分离卷积（严格控制stride和padding，确保尺寸可控）"""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            # 深度卷积（分组卷积，groups=c1）
            nn.Conv2d(c1, c1, k, s, autopad(k), groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            # 点卷积（1x1，调整通道数）
            nn.Conv2d(c1, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------------
# 通道注意力模块（轻量化 SE）
# ----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化（不改变批次和通道数）
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()  # 输出通道权重（0-1）
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))  # 计算通道注意力权重
        return x * w  # 逐通道加权（不改变尺寸）

# ----------------------------
# 主干模块：轻量化 + 通道注意力（支持可控下采样）
# ----------------------------
class LightMB_CA(nn.Module):
    def __init__(self, c1, c2, n=2, kernel_size=3, e=0.5, p=1.0, stride=1):
        """
        轻量化多分支模块（带通道注意力）
        参数：
            stride: 下采样控制（1=不下采样，2=下采样一次），由外部stage传入
        """
        super().__init__()
        # 参数验证
        assert 0 < e <= 1, "扩展比例e应该在(0,1]范围内"
        assert 0 < p <= 1, "隐藏层比例p应该在(0,1]范围内"
        assert stride in (1, 2), "stride必须为1或2（控制是否下采样）"
        
        self.c = int(c2 * e)  # 中间通道数（轻量化设计）
        self.stride = stride  # 保存下采样参数，用于残差适配
        
        # 初始卷积：集中控制下采样（仅此处可能stride=2）
        self.cv_first = Conv(c1, 2 * self.c, 1, s=stride)  # 关键：stride由外部传入
        
        # 分支1：深度卷积（stride=1，不改变尺寸）
        self.cv_block_1 = DWConv(2 * self.c, self.c, k=3, s=1)  # 固定stride=1
        
        # 分支2：瓶颈结构（所有层stride=1，不改变尺寸）
        dim_hid = int(p * 2 * self.c)  # 隐藏层通道数
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),  # 升维/降维（stride=1）
            DWConv(dim_hid, dim_hid, kernel_size, 1),  # 深度卷积（固定stride=1）
            Conv(dim_hid, self.c, 1, 1)  # 调整回目标通道（stride=1）
        )
        
        # 序列深度卷积块（均为stride=1，不改变尺寸）
        self.blocks = nn.ModuleList([
            DWConv(self.c, self.c, k=kernel_size, s=1) for _ in range(n)  # 固定stride=1
        ])
        
        # 通道注意力模块（不改变尺寸）
        self.ca = ChannelAttention(self.c)
        
        # 最终融合卷积（1x1，调整通道至c2，stride=1）
        self.cv_final = Conv((4 + n) * self.c, c2, 1, s=1)  # 固定stride=1
        
        # 残差连接适配：确保尺寸和通道与主分支匹配
        if c1 != c2 or stride != 1:
            self.shortcut = Conv(c1, c2, 1, s=stride, act=False)  # 同步下采样和通道调整
        else:
            self.shortcut = nn.Identity()  # 通道和尺寸一致时直接残差

    def forward(self, x):
        # 残差分支（与主分支尺寸/通道保持一致）
        residual = self.shortcut(x)  # 若stride=2，残差尺寸同步减半
        
        # 初始特征提取（若stride=2，此处尺寸减半）
        y = self.cv_first(x)  # y的尺寸 = x的尺寸 / stride
        
        # 多分支处理（所有分支尺寸与y一致，因内部stride=1）
        y0 = self.cv_block_1(y)  # 分支1：深度卷积（尺寸不变）
        y1 = self.cv_block_2(y)  # 分支2：瓶颈结构（尺寸不变）
        y2, y3 = y.chunk(2, 1)   # 分支3：通道拆分（尺寸与y相同）
        
        # 通道注意力分支（尺寸不变）
        y_main = self.ca(y3)
        
        # 收集所有分支特征（尺寸均为y的尺寸）
        outs = [y0, y1, y2, y_main]
        
        # 序列处理（尺寸不变）
        current = y_main
        for block in self.blocks:
            current = block(current)
            outs.append(current)
        
        # 特征融合 + 残差连接（确保尺寸匹配）
        return self.cv_final(torch.cat(outs, 1)) + residual

# ----------------------------
# 测试主程序：验证下采样功能
# ----------------------------
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试1：stride=1（不下采样）
    batch_size, in_channel, height, width = 1, 256, 40, 40  # 模拟stage3输出（40x40）
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    model_stride1 = LightMB_CA(c1=256, c2=512, n=1, kernel_size=5, e=0.5, p=0.8, stride=1).to(device)
    outputs_stride1 = model_stride1(inputs)
    print(GREEN + f"stride=1测试: 输入尺寸{inputs.shape[2:]} → 输出尺寸{outputs_stride1.shape[2:]}" + RESET)
    assert outputs_stride1.shape[2:] == (40, 40), "stride=1时尺寸应保持不变"
    
    # 测试2：stride=2（下采样一次）
    model_stride2 = LightMB_CA(c1=256, c2=512, n=1, kernel_size=5, e=0.5, p=0.8, stride=2).to(device)
    outputs_stride2 = model_stride2(inputs)
    print(GREEN + f"stride=2测试: 输入尺寸{inputs.shape[2:]} → 输出尺寸{outputs_stride2.shape[2:]}" + RESET)
    assert outputs_stride2.shape[2:] == (20, 20), "stride=2时尺寸应减半"
    
    # 计算 FLOPs
    print(ORANGE + "\n计算FLOPs（stride=2场景）:" + RESET)
    flops, macs, _ = calculate_flops(
        model=model_stride2,
        input_shape=(batch_size, in_channel, height, width),
        output_as_string=True,
        output_precision=4,
        print_detailed=True
    )
    print(RESET)
