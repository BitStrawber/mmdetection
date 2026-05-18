# Copyright (c) OpenMMLab. All rights reserved.
"""
ResNet + RFTM (Residual Feature Transference Module)
参考论文: Learning Heavily-Degraded Prior for Underwater Object Detection (TCSVT 2023)

RFTM是一个轻量特征增强模块，插入在ResNet的layer1和layer2之间。
它学习将水下重度退化区域的特征映射到"检测友好"的特征空间。
"""
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from .resnet import ResNet


class RFTM(BaseModule):
    """Residual Feature Transference Module

    结构: MaxPool(2,stride=2) + 3×Conv3×3(ReLU on first two)
    放置在ResNet backbone中，增强水下退化区域的特征表达。

    Args:
        in_channels (int): 输入通道数 (layer1输出通道, default=256)
        init_cfg (dict): 初始化配置
    """

    def __init__(self, in_channels=256, init_cfg=None):
        super().__init__(init_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = build_conv_layer(
            None, in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            None, in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = build_conv_layer(
            None, in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 上采样恢复空间分辨率（MaxPool 2x → Upsample 2x）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): 输入特征 (B, C, H, W)

        Returns:
            Tensor: 增强后的特征 (B, C, H, W)，与输入尺寸相同
        """
        identity = x
        x = self.maxpool(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        x = self.upsample(x)
        # 如果上采样后尺寸不完全匹配，做插值
        if x.shape[-2:] != identity.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=identity.shape[-2:], mode='bilinear', align_corners=False)
        return x + identity


@MODELS.register_module()
class ResNetWithRFTM(ResNet):
    """ResNet backbone with RFTM plugin between layer1 and layer2.

    在标准ResNet中插入RFTM模块，增强水下特征提取能力。
    支持加载torchvision预训练权重（仅ResNet部分）。

    Args:
        depth (int): ResNet depth, from {50, 101, 152}
        rftm_channels (int): RFTM模块的通道数 (应与layer1输出匹配)
        **kwargs: 其他ResNet参数
    """

    def __init__(self, depth=50, rftm_channels=256, **kwargs):
        super().__init__(depth=depth, **kwargs)
        # 在layer1之后插入RFTM
        self.rftm = RFTM(in_channels=rftm_channels)
        # 标记RFTM插入位置
        self.with_rftm = True

    def forward(self, x):
        """Forward function with RFTM enhancement."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            # 在layer1 (i=0) 之后应用RFTM
            if i == 0 and self.with_rftm:
                x = self.rftm(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
