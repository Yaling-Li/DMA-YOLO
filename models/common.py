# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules 网络组件模块
"""

import logging
import math
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional
from torch.cuda import amp

#这部分为util的内容
from utils.datasets import exif_transpose, letterbox
from utils.general import (colorstr, increment_path, make_divisible, non_max_suppression, save_one_box, scale_coords,
                           xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

from models.GhostV2 import Ghostblockv2

LOGGER = logging.getLogger(__name__)

def autopad(k, p=None):  # kernel, padding
    """# Pad to 'same'
    # 为same卷积或same池化自动扩充
    # 通过卷积核的大小来计算需要的padding为多少才能把tensor补成原来的形状。
    用于Conv函数和Classify函数中
       根据卷积核大小k自动计算卷积核padding数（0填充）
       v5中只有两种卷积：
          1、下采样卷积:conv3x3 s=2 p=k//2=1
          2、feature size不变的卷积:conv1x1 s=1 p=k//2=1
       :params k: 卷积核的kernel_size
       :return p: 自动计算的需要pad值（0填充）

    # 如果k是int 则进行k//2 若不是则进行x//2"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # 架构图中的标准卷积： conv+BN+act
    """在Focus、Bottleneck、BottleneckCSP、C3、SPP、DWConv、TransformerBloc等模块中调用
          Standard convolution  conv+BN+act
          :params c1: 输入的channel值
          :params c2: 输出的channel值
          :params k: 卷积的kernel_size
          :params s: 卷积的stride
          :params p: 卷积的padding  一般是None  可以通过autopad自行计算需要pad的padding数
          :params g: 卷积的groups数  =1就是普通的卷积  >1就是深度可分离卷积
          :params act: 激活函数类型   True就是SiLU()/Swish   False就是不使用激活函数
                       类型是nn.Module就使用传进来的激活函数类型
          """
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # conv调用nn.Conv2d函数，p采用autopad，不使用偏置bias=False 因为下面做融合时，这个卷积的bias会被消掉， 所以不用
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # 如果act是true则使用nn.SiLU 否则不使用激活函数
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):# 前向计算，网络执行的顺序是根据forward函数来决定的
        return self.act(self.bn(self.conv(x)))# 先conv卷积然后在bn最后act激活

    def forward_fuse(self, x):# 前向融合计算
        #用于Model类的fuse函数,融合conv+bn 加速推理 一般用于测试/验证阶段
        return self.act(self.conv(x))# 这里只有卷积和激活

class DWConv(Conv):
    # Depth-wise convolution class# 这里的深度可分离卷积，主要是将通道按输入输出的最大公约数进行切分，在不同的特征图层上进行特征学习
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Focus(nn.Module):
    # 基本组件,模型的一开始，将输入图像先 slice 成4份，再做concat
    # 设计思路：理论上从高分辨率图像中，周期性的抽出像素点重构到低分辨率图像中，即将图像相邻的四个位置进行堆叠，聚焦wh维度信息到c通道空，提高每个点感受野，并减少原始信息的丢失。这个组件并不是为了增加网络的精度的，而是为了减少计算量，增加速度。
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)#concat后的卷积
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2) 有点像做一个下采样
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Bottleneck(nn.Module):
    # 标准Bottlenck 网络架构中的bottlenack模块 ，分为True和False残差块结构   由一些 1x1conv、3x3conv、残差块组成
    """在BottleneckCSP和yolo.py的parse_model中调用
            Standard bottleneck  Conv+Conv+shortcut
            :params c1: 第一个卷积的输入channel
            :params c2: 第二个卷积的输出channel
            :params shortcut: bool 是否有shortcut连接 默认是True
            :params g: 卷积分组的个数  =1就是普通卷积  >1就是深度可分离卷积
            :params e: expansion ratio  e*c2就是第一个卷积的输出channel=第二个卷积的输入channel
            """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # 几个标准Bottleneck的堆叠+几个标准卷积层
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        # 叠加n次Bottleneck
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(nn.Module):
    # C3模块和BottleneckCSP模块类似，但是少了一个Conv模块 一种简化版的BottleneckCSP，因为除了Bottleneck部分只有3个卷积，可以减少参数，所以取名C3
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """在C3TR模块和yolo.py的parse_model模块调用
               CSP Bottleneck with 3 convolutions
               :params c1: 整个BottleneckCSP的输入channel
               :params c2: 整个BottleneckCSP的输出channel
               :params n: 有n个Bottleneck
               :params shortcut: bool Bottleneck中是否有shortcut，默认True
               :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
               :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
               """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        # 实验性 crossconv

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3TR(C3):
    # C3 module with TransformerBlock() 这部分是根据上面的C3结构改编而来的将原先的Bottleneck替换为调用TransformerBlock模块
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)

class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_//32, n)

class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)

class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class SPP(nn.Module):
    # SPP的提出就是为了解决CNN输入图像大小必须固定的问题
    # 融合了局部与全局特征，同时兼顾了避免裁剪图像的作用
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)# 1*1的卷积
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class ASPP(nn.Module):
    # Atrous Spatial Pyramid Pooling (ASPP) layer
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.m = nn.ModuleList([nn.Conv2d(c_, c_, kernel_size=3, stride=1, padding=(x-1)//2, dilation=(x-1)//2, bias=False) for x in k])
        self.cv2 = Conv(c_ * (len(k) + 2), c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x]+ [self.maxpool(x)] + [m(x) for m in self.m] , 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class ChannelAttentionModule(nn.Module):
    """
            :params: in_planes 输入模块的feature map的channel
            :params: ratio 降维/升维因子
            通道注意力则是将一个通道内的信息直接进行全局处理，容易忽略通道内的信息交互
            """

    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化，是取整个channel所有元素的均值 [3,5,5] => [3,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化，是取整个channel所有元素的最大值[3,5,5] => [3,1,1]
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    # 对空间注意力来说，由于将每个通道中的特征都做同等处理，容易忽略通道间的信息交互
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        # 这里要保持卷积后的feature尺度不变，必须要padding=kernel_size//2
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # 输入x = [b, c, 56, 56]
        avgout = torch.mean(x, dim=1, keepdim=True)  # avg_out = [b, 1, 56, 56]  求x的每个像素在所有channel相同位置上的平均值
        maxout, _ = torch.max(x, dim=1, keepdim=True)  # max_out = [b, 1, 56, 56]  求x的每个像素在所有channel相同位置上的最大值
        out = torch.cat([avgout, maxout], dim=1)  # x = [b, 2, 56, 56]  concat操作
        out = self.sigmoid(self.conv2d(out))  # x = [b, 1, 56, 56]  卷积操作，融合avg和max的信息，全方面考虑
        return out

class CBAM(nn.Module):
    def __init__(self, c1, c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class TransformerLayer(nn.Module):
    """
         这部分相当于原论文中的单个Encoder部分(只移除了两个Norm部分, 其他结构和原文中的Encoding一模一样)
        """
    def __init__(self, c, num_heads):
        super().__init__()
 
        self.ln1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4*c, bias=False)
        self.fc2 = nn.Linear(4*c, c, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)
 
    def forward(self, x):
        x_ = self.ln1(x)
        x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_))[0]) + x #残差
        x_ = self.ln2(x)#归一化
        x_ = self.fc2(self.dropout(self.act(self.fc1(x_))))
        x = x + self.dropout(x_)
        return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    # 这部分相当于原论文中的Encoders部分 只替换了一些编码方式和最后Encoders出来数据处理方式
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)

class Contract(nn.Module):
    # Contract函数改变输入特征的shape，将feature map的w和h维度(缩小)的数据收缩到channel维度上(放大)。如：x(1,64,80,80) to x(1,256,40,40)。
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

class Expand(nn.Module):
    # 会用在yolo.py的parse_model模块（用的不多）
    # Expand函数也是改变输入特征的shape，不过与Contract的相反， 是将channel维度(变小)的数据扩展到W和H维度(变大)。如：x(1,64,80,80) to x(1,16,160,160)。
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerLayer(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c, num_heads, window_size=7, shift_size=0, 
                mlp_ratio = 4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if num_heads > 10:
            drop_path = 0.1
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(c)
        self.attn = WindowAttention(
            c, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(c)
        mlp_hidden_dim = int(c * mlp_ratio)
        self.mlp = Mlp(in_features=c, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = ( (0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, torch.tensor(-100.0)).masked_fill(attn_mask == 0, torch.tensor(0.0))
        return attn_mask

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.permute(0, 3, 2, 1).contiguous() # [b,h,w,c]

        attn_mask = self.create_mask(x, h, w) # [nW, Mh*Mw, Mh*Mw]
        shortcut = x
        x = self.norm1(x)
        
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            # print(f"shift size: {self.shift_size}")
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = window_partition(shifted_x, self.window_size) # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c) # [nW*B, Mh*Mw, C]

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # [B, H', W', C]
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x = x.permute(0, 3, 2, 1).contiguous()
        return x # (b, self.c2, w, h)

class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.tr = nn.Sequential(*(SwinTransformerLayer(c2, num_heads=num_heads, window_size=window_size,  shift_size=0 if (i % 2 == 0) else self.shift_size ) for i in range(num_layers)))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.tr(x)
        return x

class Concat(nn.Module):
    # 这个函数是讲自身（a list of tensors）按照某个维度进行concat，常用来合并前后两个feature map，也就是上面yolov5s结构图中的Concat。
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class GhostConv(nn.Module):
    """2020提出的新型的轻量化网络模块Ghost，该模块不能增加MAP，但是可以大大减少模型的计算量
    可以用GhostConv代替一般的Conv，GhostBottleneck代替C3，至于在哪些位置代替，可以自己决定。
    Ghost Convolution https://github.com/huawei-noah/ghostnet
    Ghost Convolution 幻象卷积  轻量化网络卷积模块
    论文: https://arxiv.org/abs/1911.11907
    源码: https://github.com/huawei-noah/ghostnet
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        # 第一步卷积: 少量卷积, 一般是一半的计算量
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        # 第二步卷积: cheap operations 使用3x3或5x5的卷积, 并且是逐个特征图的进行卷积（Depth-wise convolutional）
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        # 注意, 源码中并不是直接Identity连接, 而是先经过一个DWConv + Conv, 再进行shortcut连接的。
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class AutoShape(nn.Module):
    #这个模块是一个模型扩展模块，给模型封装成包含前处理、推理、后处理的模块(预处理 + 推理 + nms)，用的不多。
    """在yolo.py中Model类的autoshape函数中使用
        将model封装成包含前处理、推理、后处理的模块(预处理 + 推理 + nms)  也是一个扩展模型功能的模块
        autoshape模块在train中不会被调用，当模型训练结束后，会通过这个模块对图片进行重塑，来方便模型的预测
        自动调整shape，我们输入的图像可能不一样，可能来自cv2/np/PIL/torch 对输入进行预处理 调整其shape，
        调整shape在datasets.py文件中,这个实在预测阶段使用的,model.eval(),模型就已经无法训练进入预测模式了
        input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
        """
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        #开启验证模式
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model.model[-1]  # Detect()
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # 这里的imgs针对不同的方法读入，官方也给了具体的方法，size是图片的尺寸，就比如最上面图片里面的输入608*608*3
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        # 图片如果是tensor格式 说明是预处理过的, 直接正常进行前向推理即可 nms在推理结束进行(函数外写)
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process 图片不是tensor格式 就先对图片进行预处理  Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference 预处理结束再进行前向推理  Inference
            y = self.model(x, augment, profile)[0]  # forward 前向推理
            t.append(time_sync())

            # Post-process 前向推理结束后 进行后处理Post-process  nms
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])# 将nms后的预测结果映射回原图尺寸

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)
#用的极少
class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n

class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    """
            这是一个二级分类模块, 比如做车牌的识别, 先识别出车牌, 如果想对车牌上的字进行识别, 就需要二级分类进一步检测.
            如果对模型输出的分类再进行分类, 就可以用这个模块. 不过这里这个类写的比较简单, 若进行复杂的二级分类, 可以根据自己的实际任务可以改写, 这里代码不唯一.
            Classification head, i.e. x(b,c1,20,20) to x(b,c2)
            用于第二级分类   可以根据自己的任务自己改写，比较简单
            """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1) 自适应平均池化操作
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()# 展平

    def forward(self, x):
        # 先自适应平均池化操作，然后拼接
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)# 对z进行展平操作

#---------------------------------------新自适应融合模块----------------------------------------
class AdaptADD(nn.Module):
    # 这个函数是讲自身（a list of tensors）按照某个维度进行concat，常用来合并前后两个feature map，也就是上面yolov5s结构图中的Concat。
    # Concatenate a list of tensors along dimension
    def __init__(self, level, out_ch, dimension, dim1, dim2, dim3=1, rfb=False):
        super().__init__()
        self.level= level
        self.d = dimension
        self.dims = [dim1, dim2, dim3]
        compress_c = 8 if rfb else 16
        self.compress_level = add_conv(self.dims[2], self.dims[0], 1, 1)#第三层特征降维
        self.weight_map = add_conv(self.dims[0], compress_c, 1, 1)# 计算权重
        self.weight_levels = nn.Conv2d(compress_c * level, level, kernel_size=1, stride=1, padding=0)
        self.expand = add_conv(self.dims[0], out_ch, 3, 1)

    def forward(self, x):
        weights = []
        weight0 = self.weight_map(x[0])
        weight1 = self.weight_map(x[1])
        weights.append(weight0)
        weights.append(weight1)

        if self.level == 3:
            map = self.compress_level(x[2])
            weight2 = self.weight_map(map)
            weights.append(weight2)

        weights = torch.cat(weights, self.d) # 将各个权重矩阵按照通道concat
        #print (weights.shape)
        weights = self.weight_levels(weights)
        levels_weight = F.softmax(weights, dim=1)

        if self.level == 2:
            fused_out_reduced = x[0] * levels_weight[:, 0:1, :, :] + x[1] * levels_weight[:, 1:, :, :]
        else:
            fused_out_reduced = x[0] * levels_weight[:, 0:1, :, :] + x[1] * levels_weight[:, 1:2, :, :] + map * levels_weight[:, 2:, :, :]

        fused_out_reduced = self.expand(fused_out_reduced)

        return  fused_out_reduced

class AdaptConcat(nn.Module):
    # 这个函数是讲自身（a list of tensors）按照某个维度进行concat，常用来合并前后两个feature map，也就是上面yolov5s结构图中的Concat。
    # Concatenate a list of tensors along dimension
    def __init__(self, level, dimension, dim1, dim2, dim3=1, rfb=False):
        super().__init__()
        self.level= level
        self.d = dimension
        self.dims = [dim1, dim2, dim3]
        compress_c = 8 if rfb else 16
        #self.compress_level = add_conv(self.dims[2], self.dims[0], 1, 1)#第三层特征降维
        self.weight_map0 = add_conv(self.dims[0], compress_c, 1, 1)# 计算权重
        self.weight_map1 = add_conv(self.dims[1], compress_c, 1, 1)  # 计算权重
        self.weight_map2 = add_conv(self.dims[2], compress_c, 1, 1)  # 计算权重
        self.weight_levels = nn.Conv2d(compress_c * level, level, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        weights = []
        weight0 = self.weight_map0(x[0])
        weight1 = self.weight_map1(x[1])
        weights.append(weight0)
        weights.append(weight1)

        if self.level == 3:
            weight2 = self.weight_map2(x[2])
            weights.append(weight2)

        weights = torch.cat(weights, self.d) # 将各个权重矩阵按照通道concat
        #print (weights.shape)
        weights = self.weight_levels(weights)
        levels_weight = F.softmax(weights, dim=1)

        x0 = x[0] * levels_weight[:, 0:1, :, :]
        x1 = x[1] * levels_weight[:, 1:2, :, :]
        fused_out_reduced = torch.cat((x0, x1), 1)

        if self.level == 3:
            x2 = x[2] * levels_weight[:, 2:, :, :]
            fused_out_reduced = torch.cat((x0, x1, x2), 1)

        return  fused_out_reduced

class AdConcat2(nn.Module):
    # 结合BiFPN 设置可学习参数 学习不同分支的权重
    # 两个分支concat操作
    def __init__(self, dimension=1):
        super(AdConcat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)

class AdConcat3(nn.Module):
    # 三个分支concat操作
    def __init__(self, dimension=1):
        super(AdConcat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)

class Adapt_Add2(nn.Module):
    # 结合BiFPN 设置可学习参数 学习不同分支的权重
    # 两个分支add操作
    def __init__(self):
        super(Adapt_Add2, self).__init__()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        return self.silu(weight[0] * x[0] + weight[1] * x[1])

class Adapt_Add3(nn.Module):
    # 三个分支add操作
    def __init__(self, d1, d2, d3):
        super(Adapt_Add3, self).__init__()

        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

        self.conv = nn.Conv2d(d1, d3, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        return self.silu(weight[0] * self.conv(x[0]) + weight[1] * self.conv(x[1]) + weight[2] * x[2])

def add_conv(in_ch, out_ch, ksize=1, stride=1):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage

class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        # 输入的三个特征层的channels, 根据实际修改， 调整到一样的尺度
        self.dim = [512, 256, 256]# 输入的三层特征，通道数分别为512，256，256
        self.inter_dim = self.dim[self.level] # 中间输出通道数，得到第几层的融合结果，输出通道数就是该层的通道数
        # 每个层级三者输出通道数需要一致
        if level==0: #第0层需要其他两层经过下采样 输入通道分别为自身通道数，输出通道为第0层的通道数， 卷积核大小为3， 步长为2
            self.stride_level_1 = add_conv(self.dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            if self.dim[1] != self.dim[2]:
                self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis

	# 尺度大小 level_0 < level_1 < level_2
    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            if self.dim[1] != self.dim[2]:
                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            else:
                level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)   # alpha等产生

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

#--------------------------------------------加入注意力模块----------------------------------------

class CoorAttention(nn.Module):
    """
    CA Coordinate Attention 协同注意力机制
    论文 CVPR2021: https://arxiv.org/abs/2103.02907
    源码: https://github.com/Andrew-Qibin/CoordAttention/blob/main/coordatt.py
    CA注意力机制是一个Spatial Attention 相比于SAM的7x7卷积, CA建立了远程依赖
    """

    def __init__(self, c1, c2, reduction=32):
        super(CoorAttention, self).__init__()
        # [B, C, H, W] -> [B, C, H, 1]
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # 自适应平均池化，指定输出（H,W），x轴即为水平方向w，进而使w的值变为1
        # [B, C, H, W] -> [B, C, 1, W]
        self.pool_w= nn.AdaptiveAvgPool2d((1, None)) #在y轴进行平均池化操作，y轴为垂直方向h，进而使h的值变为1

        c_ = max(8, c1 // reduction)   # 对中间层channel做一个限制 不得少于8
        # 将x轴信息和y轴信息融合在一起
        self.conv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.Hardswish()  # 这里自己可以实验什么激活函数最佳 论文里是hard-swish
        #self.act = nn.ReLU()

        self.conv_w = nn.Conv2d(c_, c2, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv2d(c_, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # [B, C, H, W] -> [B, C, H, 1]
        x_h = self.pool_h(x)   # h avg pool
        # [B, C, H, W] -> [B, C, 1, W] -> [B, C, W, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # w avg pool

        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # split  x_h: [B, C, H, 1]  x_w: [B, C, W, 1]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # [B, C, W, 1] -> [B, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 基于W和H方向做注意力机制 建立远程依赖关系
        out = identity * a_w * a_h

        return out

class CABottleneck(nn.Module):
    """
           Standard bottleneck  Conv+Conv+shortcut
            :params c1: 第一个卷积的输入channel
            :params c2: 第二个卷积的输出channel
            :params shortcut: bool 是否有shortcut连接 默认是True
            :params g: 卷积分组的个数  =1就是普通卷积  >1就是深度可分离卷积
            :params e: expansion ratio  e*c2就是第一个卷积的输出channel=第二个卷积的输入channel
            """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, reduction=32):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.ca = CoorAttention(c2, c2, reduction)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.ca(self.cv2(self.cv1(x))) if self.add else self.ca(self.cv2(self.cv1(x)))

class C3CA(C3):
    # C3 module with CABottleneck()
    # 将CA加入到C3中
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class SPPFCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

#----------------------------------------------修改卷积块----------------------------------------
class SCConv(nn.Module):
    # 利用自校正卷积取代原来特征提取网络中的常规卷积  解决小目标问题（2020）
    # 小目标由于携带信息少导致特征表达能力较弱，经过多层次的卷积操作后能提取到的特征较少，因此检测困难。利用自校正卷积取代特征提取网络中的常规卷积，以扩展感受野丰富输出，进而强化对弱特征的提取能力。
    # http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
    # https://github.com/MCG-NKU/SCNet
    # 在不调整模型架构的情况下改进CNN的基本卷积特征转换过程。提出了一种新颖的自校正卷积，该卷积可以通过特征的内在通信达到扩增卷积感受野的目的，进而增强输出特征的多样性。不同于标准卷积采用小尺寸核（例如3×3
    #卷积）同时融合空间维度域与通道维度的信息，SCConv可以通过自校正操作自适应地在每个空间位置周围建立了远程空间和通道间依存关系。因此，它可以帮助CNN生成更具判别能力的特征表达，因其具有更丰富的信息。

    def __init__(self, c1, c2, stride, groups=1, dilation=1, pooling_r=4):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(c1, c1, kernel_size=3, stride=1,
                                padding=autopad(3, None), dilation=dilation,
                                groups=groups, bias=False),
                    nn.BatchNorm2d(c1),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(c1, c1, kernel_size=3, stride=1,
                                padding=autopad(3, None), dilation=dilation,
                                groups=groups, bias=False),
                    nn.BatchNorm2d(c1),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(c1, c2, kernel_size=3, stride=stride,
                                padding=autopad(3, None), dilation=dilation,
                                groups=groups, bias=False),
                    nn.BatchNorm2d(c2),
                    )
    def forward(self, x):
        identity = x
        # F.interpolate 利用插值的方式进行数组采样，输出空间的大小（k2先下采样，然后再利用双线性插值上采样）
        y_ = F.interpolate(self.k2(x), identity.size()[2:])
        y_ = torch.add(identity, y_)
        out = torch.sigmoid(y_) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4
        return out

class GnConv(nn.Module):  # gnconv模块
    # 递归门卷积（性能优于swin transformer和convnext）
    # CNN具有平移不变性和局部性，缺乏全局建模长距离建模能力，引入自然语言处理领域的框架Transformer来形成CNN+Transformer架构
    def __init__(self, c1, c2, ksize=1, stride=1, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [c1 // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(c1, 2 * c1, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = Conv(c1, c2, ksize, stride)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s

        #print('[gconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

class HorBlock(nn.Module):# HorBlock模块
    r""" HorNet block yoloair
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=GnConv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = GnConv(dim, dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape # [512]
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))
        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)

        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
# 使用C3HB会造成梯度消失问题(减少重复次数)
class C3HB(nn.Module):
    # CSP HorBlock with 3 convolutions by iscyy/yoloair
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(HorBlock(c_) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3GhostV2(C3):
    # C3GV2 module with GhostV2(for iscyy)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c1_ = 16
        self.c2_ = 16 * e
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Ghostblockv2(c_, self.c1_, c_) for _ in range(n)))

# spd(2022)替换自校正卷积(2020年)
class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

class SM(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

"""class SMMConv(nn.Module):
    def __init__(self, c1, c2):
        super(SMMConv, self).__init__()
        c_ = int(c2/2)
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c2, 3, 1)
        self.sm = SM()
        self.mp  = MP()
    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        x_2 = self.sm(x_2)
        return torch.cat([x_2, x_1], 1)"""
class SMMConv(nn.Module):
    def __init__(self, c1, c2):
        super(SMMConv, self).__init__()
        c_ = int(c1/2)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c1, c_, 5, 1)
        self.sm = SM()
    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        x_3 = torch.cat([x_1,x_2],1)
        x_4 = self.sm(x_3)

        return x_4
class DMMConv2(nn.Module):
    def __init__(self, c1, c2):
        super(DMMConv2, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)

        self.sm = SM()
        self.mp  = MP()
    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        x_2 = self.sm(x)

        return torch.cat([x_2, x_1], 1)

class DMMConv(nn.Module):
    def __init__(self, c1, c2):
        super(DMMConv, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 3, 1)
        self.sm = SM()
        self.mp  = MP()
    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)
        x_2 = self.cv2(x)
        x_2 = self.sm(x_2)

        return torch.cat([x_2, x_1], 1)

class DMConv(nn.Module):
    def __init__(self, c1, c2):
        super(DMConv, self).__init__()
        self.cv1 = Conv(c1, c2, 3, 1)
        self.sm = SM()
    def forward(self, x):
        x = self.cv1(x)
        x = self.sm(x)

        return x

class DMMixConv2d(nn.Module):
    # MixConv2d 混合深度卷积就是使用不同大小的卷积核对深度卷积的不同channel分组处理。也可以看作是 分组深度卷积 + Inception结构 的多种卷积核混用。
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        """
                :params c1: 输入feature map的通道数
                :params c2: 输出的feature map的通道数（这个函数的关键点就是对c2进行分组）
                :params k: 混合的卷积核大小 其实论文里是[3, 5, 7...]用的比较多的
                :params s: 步长 stride
                :params equal_ch: 通道划分方式 有均等划分和指数划分两种方式  默认是均等划分"""
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group 均等划分通道
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group 指数划分通道
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))
        #return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))#

class BAM(C3):
    # C3 module with CABottleneck()
    # 将CA加入到C3中
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))