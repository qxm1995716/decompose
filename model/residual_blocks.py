import math
import torch.nn as nn
import torch
from typing import Optional, Any, Dict, List, Type

from scipy.sparse import diags

# Code adapted from timm (PyTorch Image Models)
# Source: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
# License: Apache 2.0
# Original Copyright 2019, Ross Wightman (and other contributors as noted in the source file)

class ResBasicBlock(nn.Module):
    r"""
    This block does not consider any kinds of down sample operations.
    """
    expansion = 1
    def __init__(self,
                 in_channels,
                 channels,
                 ksize=3,
                 stride=1,
                 downsample: Optional[nn.Module] = None,
                 dilation=1,
                 channel_reduce_factor=1,
                 act_func=nn.ReLU,
                 norm_func=nn.BatchNorm2d,
                 drop_block=None,
                 drop_path=None):
        super(ResBasicBlock, self).__init__()
        intermediate_channel = channels // channel_reduce_factor  # 此处是维度控制参数，其作为第一层卷积的输出维度
        out_channels = channels * self.expansion  # 输入维度，默认expansion为1，该参数主要是在bottleneck使用
        # 基础模块
        self.conv1 = nn.Conv2d(in_channels, intermediate_channel, kernel_size=ksize, stride=stride,
                               padding=ksize // 2, dilation=dilation, bias=False)
        self.bn1 = norm_func(intermediate_channel)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_func()

        self.conv2 = nn.Conv2d(intermediate_channel, out_channels, kernel_size=ksize, stride=1,
                               padding=ksize // 2, dilation=dilation, bias=False)
        self.bn2 = norm_func(out_channels)
        self.act2 = act_func()
        # shortcut的维度变化
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.drop_path = drop_path

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.act2(x)

        return x


class ResBottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 ksize=3,
                 stride=1,
                 downsample: Optional[nn.Module] = None,
                 base_width=64,
                 dilation=1,
                 group=1,
                 channel_reduce_factor=1,
                 act_func=nn.ReLU,
                 norm_func=nn.BatchNorm2d,
                 drop_block=None,
                 drop_path=None):
        super(ResBottleneckBlock, self).__init__()
        self.base_channel = int(math.floor(channels * (base_width / 64)) * group)
        self._f_channel = self.base_channel // channel_reduce_factor
        self.dilation = dilation

        self.conv1 = nn.Conv2d(in_channels, self._f_channel, kernel_size=ksize, bias=False)
        self.bn1 = norm_func(self._f_channel)
        self.act1 = act_func()

        self.conv2 = nn.Conv2d(self._f_channel, self.base_channel, kernel_size=ksize, stride=stride, padding=dilation,
                               groups=group, bias=False)
        self.bn2 = norm_func(self.base_channel)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_func()

        self.conv3 = nn.Conv2d(self._f_channel, channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_func(channels)
        self.act3 = act_func()

        self.downsample = downsample if downsample is not None else nn.Identity()
        self.drop_path = drop_path

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = x + self.downsample(shortcut)
        x = self.act3(x)

        return x

# 此处暂时不考虑通过残差模块来完成下采样过程，因此downsample只负责维度变化
def channel_reduction(input_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel)
    )


class ResModule(nn.Module):
    def __init__(self, in_channel, out_channel, ksize, num_block, block_func, stride, act_func=nn.LeakyReLU,
                 bn_func=nn.BatchNorm2d):
        super(ResModule, self).__init__()
        backbone = []
        in_d = in_channel
        for idx in range(num_block):
            down_sampler = None
            if in_d != out_channel * block_func.expansion or stride != 1:
                down_sampler = channel_reduction(in_d, out_channel * block_func.expansion)
            backbone.append(block_func(in_channels=in_d, channels=out_channel, ksize=ksize, stride=stride,
                                       downsample=down_sampler, act_func=act_func, norm_func=bn_func))
            in_d = out_channel * block_func.expansion
            stride = 1  # 在第一个block后步长设定为1
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        return self.backbone(x)
