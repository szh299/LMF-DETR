"""     
融合版HGNetv2主干网络：动态融合LightMB_CA（轻量化）与iRMB_DualAtt（强特征）
reference:
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py     
Copyright (c) 2024 The D-FINE Authors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re
from .common import FrozenBatchNorm2d  # 冻结批归一化层（用于预训练权重加载）
from ..core import register  # 框架注册机制，使网络可被自动识别
from ..extre_module.custom_nn.attention.ema import EMA  # EMA注意力（iRMB_DualAtt内部复用）
from ..extre_module.custom_nn.attention.simam import SimAM  # 备用注意力模块
from ..misc.dist_utils import Multiprocess_sync  # 分布式训练同步工具
from .hgnetv2 import HGNetv2, HG_Stage  # 继承原始HGNetv2的主干和Stage基类
from functools import partial  # 用于部分参数绑定

# 导入自定义创新模块
from ..extre_module.innovate.iRMB_DualAtt import iRMB_DualAtt  # 双重注意力模块（EMA空间+ECA通道）
from ..extre_module.innovate.LightMB_CA import LightMB_CA  # 轻量化多分支+通道注意力模块


# 初始化工具函数（继承自原始HGNetv2）
kaiming_normal_ = nn.init.kaiming_normal_  # Kaiming初始化
zeros_ = nn.init.zeros_  # 零初始化
ones_ = nn.init.ones_    # 全1初始化

__all__ = ['HGNetv2_LightMB_CA_iRMB_DualAtt']  # 导出网络类，供框架自动注册


@register()  # 注册为可识别的Stage模块
class HG_Stage_LightMB_CA_iRMB_DualAtt(HG_Stage):
    """混合Stage模块：动态融合LightMB_CA与iRMB_DualAtt
    
    设计思路：
    - 浅层stage（light_block=False）：使用iRMB_DualAtt，通过双重注意力增强细节特征
    - 深层stage（light_block=True）：使用LightMB_CA，通过轻量化设计降低计算量，保留通道注意力
    """
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3,
                 use_lab=False, agg='se', drop_path=0.):
        # 调用父类构造函数，复用原始HG_Stage的基础初始化
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num,
                         downsample, light_block, kernel_size, use_lab, agg, drop_path)

        blocks_list = []  # 存储当前stage的所有block
        for i in range(block_num):
            # 输入通道适配：第一个block匹配stage输入通道，后续匹配输出通道（保证残差兼容）
            in_c = in_chs if i == 0 else out_chs
            
            # 动态选择模块，严格控制下采样（仅第一个block可能下采样）
            if light_block:
                # 深层轻量化：LightMB_CA
                block = LightMB_CA(
                    c1=in_c,               # 输入通道
                    c2=out_chs,            # 输出通道
                    n=layer_num,           # 模块内堆叠层数
                    kernel_size=kernel_size,  # 卷积核大小
                    e=0.5,                 # 扩展因子
                    p=0.8,                 # 分支2中间维度比例
                    stride=2 if (i == 0 and downsample) else 1  # 下采样控制
                )
            else:
                # 浅层强特征：iRMB_DualAtt
                block = iRMB_DualAtt(
                    dim_in=in_c,           # 输入通道
                    dim_out=out_chs,       # 输出通道
                    exp_ratio=1.2,         # 扩展因子
                    dw_ks=kernel_size,     # 深度卷积核大小
                    stride=2 if (i == 0 and downsample) else 1,  # 下采样控制
                    drop_path=drop_path    # 随机深度概率
                )
            blocks_list.append(block)

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        # 打印block输出尺寸（仅首末block，减少冗余）
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 or i == len(self.blocks)-1:
                print(f"Stage block {i}输出尺寸: {x.shape[2:]}")  # 格式：(H, W)
        return x


@register()  # 注册为主干网络，供下游任务调用
class HGNetv2_LightMB_CA_iRMB_DualAtt(HGNetv2):
    """融合版HGNetv2主干网络：平衡性能与效率
    
    核心改进：
    - 动态选择模块：浅层用iRMB_DualAtt增强特征，深层用LightMB_CA轻量化
    - 严格控制下采样链：640→160→160→80→40→20（最终匹配400长度位置嵌入）
    - 随深度递增DropPath概率，增强过拟合抑制
    """
    # 网络架构配置（B0版本）
    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],  # 输入3→16→16（stem输出，4倍下采样）
            'stage_config': {
                # 格式：[in_c, mid_c, out_c, block_num, downsample, light_block, ksz, layer_num]
                "stage1": [16, 16, 64, 1, False, False, 3, 1],  # 不下采样（160→160）
                "stage2": [64, 32, 128, 1, True, False, 3, 1],   # 下采样（160→80）
                "stage3": [128, 64, 256, 2, True, True, 5, 1],   # 下采样（80→40）
                "stage4": [256, 128, 512, 1, True, True, 5, 1],  # 下采样（40→20）
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        }
    }

    def __init__(self, name='B0',
                 use_lab=False,
                 return_idx=(1, 2, 3),  # 返回stage1(80×80)、stage2(40×40)、stage3(20×20)
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 agg='se',
                 local_model_dir='weight/hgnetv2/'):
        # 校验架构名称
        if name not in self.arch_configs:
            raise ValueError(f"仅支持{B0}架构，输入名称: {name}")
        
        super().__init__(name, use_lab, return_idx,
                         freeze_stem_only, freeze_at,
                         freeze_norm, pretrained, agg,
                         local_model_dir)

        # 构建stages
        stage_config = self.arch_configs[name]['stage_config']
        self.stages = nn.ModuleList()
        for i, (k, v) in enumerate(stage_config.items()):
            in_c, mid_c, out_c, block_num, downsample, light_block, ksz, layer_num = v
            self.stages.append(
                HG_Stage_LightMB_CA_iRMB_DualAtt(
                    in_c, mid_c, out_c, block_num, layer_num,
                    downsample, light_block, ksz, use_lab, agg,
                    drop_path=0.05 * i  # 随深度递增DropPath概率
                )
            )

    def forward(self, x):
        # 打印输入及各阶段尺寸（调试用）
        print(f"输入图像尺寸: {x.shape[2:]}")  # 预期(640, 640)
        
        # stem处理（4倍下采样）
        x = self.stem(x)
        print(f"Stem输出尺寸: {x.shape[2:]}")  # 预期(160, 160)
        
        # 各stage特征提取
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            print(f"stage{i}输出尺寸: {x.shape[2:]}")  # 验证下采样链
            feats.append(x)
        
        # 返回指定索引的特征层
        return [feats[i] for i in self.return_idx]
    def get_module_distribution(self):
        """统计各模块的数量分布"""
        dist = {"LoGStem_Lite": 1, "iRMB_DualAtt": 0, "LightMB_CA": 0}
        for s in self.stages:
            for b in s.blocks:
                module_name = type(b).__name__
                if module_name in dist:
                    dist[module_name] += 1
        return dist

    def get_feature_info(self):
        """获取各阶段特征图的尺寸、通道数等信息"""
        input_size = 640
        feature_info, size = {}, input_size // 4  # stem输出尺寸：640/4=160
        stem_out_ch = self.arch_configs[self.name]["stem_channels"][2]
        feature_info["stem"] = {
            "size": size, "channels": stem_out_ch, "module": "LoGStem_Lite"
        }

        for stage_name, cfg in self.arch_configs[self.name]["stage_config"].items():
            _, _, out_c, _, downsample, light_block, _, _ = cfg
            if downsample:
                size //= 2  # 下采样时尺寸减半
            feature_info[stage_name] = {
                "size": size,
                "channels": out_c,
                "module": "LightMB_CA" if light_block else "iRMB_DualAtt"
            }
        return feature_info


# 快速测试B0架构
def test_b0_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- 测试 HGNetv2_Fusion B0 架构 ---")
    model = HGNetv2_Fusion(name="B0").to(device)
    x = torch.randn(1, 3, 640, 640).to(device)  # 模拟640×640输入
    outputs = model(x)
    print("输出特征图维度:")
    for i, feat in enumerate(outputs):
        print(f"  特征层{i}: {feat.shape}")  # 预期输出：(80×80), (40×40), (20×20)
    print("✓ B0架构测试完毕")


if __name__ == "__main__":
    test_b0_model()
