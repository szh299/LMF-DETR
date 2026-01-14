"""
融合版 HGNetv2 主干网络（B0架构）：LoGStem_Lite + iRMB_DualAtt + LightMB_CA 三模块协同
设计理念：
- LoGStem_Lite: 高效特征提取入口，平衡细节保留与计算效率
- iRMB_DualAtt: 浅层注意力增强模块
- LightMB_CA: 深层轻量化通道注意力模块
"""

import torch
import torch.nn as nn
from .common import FrozenBatchNorm2d
from ..core import register
from .hgnetv2 import HGNetv2, HG_Stage
from ..extre_module.innovate.LoGStem_Lite import LoGStem_Lite
from ..extre_module.innovate.iRMB_DualAtt import iRMB_DualAtt
from ..extre_module.innovate.LightMB_CA import LightMB_CA


__all__ = ["HGNetv2_Fusion"]


@register()
class HG_Stage_Fusion(HG_Stage):
    """融合 Stage 模块：浅层用 iRMB_DualAtt，深层用 LightMB_CA"""

    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3,
                 use_lab=False, agg="se", drop_path=0.):
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num,
                         downsample, light_block, kernel_size, use_lab, agg, drop_path)

        blocks = []
        for i in range(block_num):
            in_c = in_chs if i == 0 else out_chs
            # 严格控制下采样：仅第一个block且downsample=True时用stride=2，其余均为1
            stride = 2 if (i == 0 and downsample) else 1  

            if light_block:
                block = LightMB_CA(
                    c1=in_c, c2=out_chs, n=layer_num,
                    kernel_size=kernel_size, e=0.5, p=0.8, stride=stride
                )
            else:
                block = iRMB_DualAtt(
                    dim_in=in_c, dim_out=out_chs,
                    exp_ratio=1.2, dw_ks=kernel_size,
                    stride=stride, drop_path=drop_path
                )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # 打印当前stage的输入尺寸及每个block的输出尺寸（精确追踪下采样点）
        print(f"Stage输入尺寸: {x.shape[2:]}")
        for i, block in enumerate(self.blocks):
            x = block(x)
            print(f"  Block {i}输出尺寸: {x.shape[2:]}")  # 定位异常缩小的block
        return x


@register()
class HGNetv2_Fusion(HGNetv2):
    """融合主干网络（仅保留B0架构，修正下采样链）"""

    arch_configs = {
        "B0": {  
            "stem_channels": [3, 16, 16],  # 输入3→16→16（stem输出，4倍下采样：640→160）
            "stage_config": {
                # 命名改为stage0-stage3，避免索引偏移；格式：[in_c, mid_c, out_c, block_num, downsample, light_block, ksz, layer_num]
                "stage0": [16, 16, 64, 1, False, False, 3, 1],  # ① 不下采样（160→160）
                "stage1": [64, 32, 128, 1, True, False, 3, 1],   # ② 下采样1次（160→80）
                "stage2": [128, 64, 256, 1, True, True, 5, 1],   # ③ 下采样1次（80→40）
                "stage3": [256, 128, 512, 1, True, True, 5, 1],  # ④ 下采样1次（40→20）
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth"
        },
    }

    def __init__(self, name="B0", use_lab=False,
                 return_idx=(1, 2, 3),  # 返回stage1(80)、stage2(40)、stage3(20)
                 freeze_stem_only=True,
                 freeze_at=0, freeze_norm=True, pretrained=False,
                 agg="se", local_model_dir="weight/hgnetv2/"):

        if name != "B0":
            raise ValueError(f"仅支持B0架构，输入名称: {name}")

        self.name = name
        super().__init__(name, use_lab, return_idx,
                         freeze_stem_only, freeze_at,
                         freeze_norm, pretrained, agg, local_model_dir)

        # 初始化stem（确保输出160×160）
        stem_cfg = self.arch_configs[name]["stem_channels"]
        self.stem = LoGStem_Lite(in_chans=stem_cfg[0], stem_dim=stem_cfg[2])

        # 初始化stages（按stage0-stage3顺序）
        stage_cfg = self.arch_configs[name]["stage_config"]
        self.stages = nn.ModuleList()
        # 按stage0到stage3的顺序加载，避免乱序
        for stage_idx in ["stage0", "stage1", "stage2", "stage3"]:
            v = stage_cfg[stage_idx]
            in_c, mid_c, out_c, block_num, downsample, light_block, ksz, layer_num = v
            self.stages.append(
                HG_Stage_Fusion(in_c, mid_c, out_c, block_num, layer_num,
                                downsample, light_block, ksz, use_lab, agg,
                                drop_path=0.05 * int(stage_idx[-1]))  # 按stage索引递增drop path
            )

    def forward(self, x):
        # 打印stem输入输出尺寸（验证stem是否正确下采样）
        print(f"输入图像尺寸: {x.shape[2:]}")  # 预期(640,640)
        x = self.stem(x)
        print(f"Stem输出尺寸: {x.shape[2:]}")  # 预期(160,160)
        
        feats = []
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            print(f"Stage{stage_idx} 输出尺寸: {x.shape[2:]}")  # 预期160→80→40→20
        return [feats[i] for i in self.return_idx]

    def get_module_distribution(self):
        dist = {"LoGStem_Lite": 1, "iRMB_DualAtt": 0, "LightMB_CA": 0}
        for s in self.stages:
            for b in s.blocks:
                module_name = type(b).__name__
                if module_name in dist:
                    dist[module_name] += 1
        return dist

    def get_feature_info(self):
        input_size = 640
        feature_info, size = {}, input_size // 4  # stem输出：640/4=160
        stem_out_ch = self.arch_configs[self.name]["stem_channels"][2]
        feature_info["stem"] = {
            "size": size, "channels": stem_out_ch, "module": "LoGStem_Lite"
        }

        for stage_idx in ["stage0", "stage1", "stage2", "stage3"]:
            cfg = self.arch_configs[self.name]["stage_config"][stage_idx]
            _, _, out_c, _, downsample, light_block, _, _ = cfg
            if downsample:
                size //= 2  # 仅downsample=True时尺寸减半
            feature_info[stage_idx] = {
                "size": size,
                "channels": out_c,
                "module": "LightMB_CA" if light_block else "iRMB_DualAtt"
            }
        return feature_info


# 快速测试B0架构（验证尺寸链）
def test_b0_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- 测试 HGNetv2_Fusion B0 架构 ---")
    model = HGNetv2_Fusion(name="B0").to(device)
    x = torch.randn(1, 3, 640, 640).to(device)  # 模拟640×640输入
    outputs = model(x)
    print("输出特征图维度:")
    for i, feat in enumerate(outputs):
        print(f"  特征层{i}: {feat.shape}")  # 预期：(80×80), (40×40), (20×20)
    # 断言验证最终特征图尺寸
    assert outputs[-1].shape[2:] == (20, 20), f"最终特征图尺寸错误，应为(20,20)，实际为{outputs[-1].shape[2:]}"
    print("✓ B0架构测试完毕")


if __name__ == "__main__":
    test_b0_model()

