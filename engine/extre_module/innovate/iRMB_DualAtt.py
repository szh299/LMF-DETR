'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ICCV2023-iRMB.png     
论文链接：https://arxiv.org/abs/2301.01146  
'''
 
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 
     
import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops  
 
import torch  
import torch.nn as nn
import torch.nn.functional as F  
import torch.nn.init as init
from timm.layers import DropPath    
from einops import rearrange

# 复用修正后的Conv模块（确保padding和stride正确）
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    """自动计算padding，确保卷积后尺寸正确"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class EMA_Attention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 仅调整通道权重，不改变尺寸
        ema_feat = self.conv(x)  # padding=kernel_size//2，stride=1 → 尺寸不变
        ema_feat = self.bn(ema_feat)
        out = self.alpha * ema_feat + (1 - self.alpha) * x
        return out * self.act(ema_feat)  # 尺寸与x一致


class ECA_Attention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 不改变空间尺寸（仅压缩H/W为1）
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # 仅调整通道权重，不改变尺寸
        y = self.avg_pool(x)  # 输出形状：(B, C, 1, 1)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))  # 1D卷积处理通道
        y = self.act(y).transpose(-1, -2).unsqueeze(-1)  # 恢复形状：(B, C, 1, 1)
        return x * y.expand_as(x)  # 尺寸与x一致


class iRMB_DualAtt(nn.Module):
    def __init__(self, dim_in, dim_out, exp_ratio=1.0, dw_ks=3, stride=1, drop_path=0.):
        super().__init__()
        dim_mid = int(dim_in * exp_ratio)
        self.norm = nn.BatchNorm2d(dim_in)
        self.act = nn.SiLU()  # 显式定义激活函数
        self.has_skip = (dim_in == dim_out and stride == 1)  # 残差连接条件

        # 1x1卷积升维（不改变尺寸）
        self.v = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_mid),
            self.act
        )

        # EMA注意力（不改变尺寸）
        self.ema_attn = EMA_Attention(dim_mid)
        
        # 深度卷积（核心下采样控制）
        self.conv_local = Conv(
            dim_mid, dim_mid, 
            k=dw_ks, 
            s=stride,  # 关键：stride由外部传入
            g=dim_mid  # 深度卷积（groups=dim_mid）
        )
        
        # ECA通道注意力（不改变尺寸）
        self.eca = ECA_Attention(dim_mid, k_size=3)
        
        # 1x1卷积降维（不改变尺寸）
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1, bias=False)
        
        # 随机深度（不改变尺寸）
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x  # 残差分支
        
        # 主干特征处理
        x = self.norm(x)  # 归一化
        x = self.v(x)     # 升维（尺寸不变）
        x = self.ema_attn(x)  # EMA注意力（尺寸不变）
        x = self.conv_local(x)  # 深度卷积（若stride=2，尺寸减半）
        x = self.eca(x)    # ECA注意力（尺寸不变）
        x = self.proj(x)   # 降维（尺寸不变）
        
        # 残差连接（确保尺寸和通道匹配）
        if self.has_skip:
            x = shortcut + self.drop_path(x)
        return x


if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, height, width = 1, 16, 32, 32  
    
    print(BLUE + "=== 测试改进版iRMB_DualAtt模块 ===" + RESET)
    out_channel = 128
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    
    # 测试stride=2（下采样）
    model_stride2 = iRMB_DualAtt(
        dim_in=in_channel, 
        dim_out=out_channel, 
        exp_ratio=1.5,
        dw_ks=5, 
        stride=2, 
        drop_path=0.05
    ).to(device)
    outputs_stride2 = model_stride2(inputs)
    print(GREEN + f'stride=2测试: 输入尺寸{inputs.shape[2:]} → 输出尺寸{outputs_stride2.shape[2:]}' + RESET)
    assert outputs_stride2.shape[2:] == (16, 16), "stride=2时尺寸应减半"
    
    # 测试stride=1（不下采样）
    model_stride1 = iRMB_DualAtt(
        dim_in=in_channel, 
        dim_out=in_channel,  # 通道一致才能用残差
        exp_ratio=1.5,
        dw_ks=5, 
        stride=1, 
        drop_path=0.05
    ).to(device)
    outputs_stride1 = model_stride1(inputs)
    print(GREEN + f'stride=1测试: 输入尺寸{inputs.shape[2:]} → 输出尺寸{outputs_stride1.shape[2:]}' + RESET)
    assert outputs_stride1.shape[2:] == (32, 32), "stride=1时尺寸应保持不变"
    
    # 计算FLOPs
    print(ORANGE + "\n计算FLOPs（stride=2场景）:" + RESET)
    flops, macs, _ = calculate_flops(
        model=model_stride2, 
        input_shape=(batch_size, in_channel, height, width),  
        output_as_string=True,
        output_precision=4,   
        print_detailed=True
    )
    print(RESET)
