# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from ..blocks import (
    _norm,
    )

from .config import ALIGN_CORNERS

from ...logger import get_logger

logger = get_logger(name=__name__)
BN_MOMENTUM = 0.1

class ModuleHelper:

    @staticmethod
    def NormAct(norm: str, num_features: int, activation: nn.Module = nn.ReLU(inplace=True), bias=True, momentum=0.1):
        return nn.Sequential(
            _norm(norm=norm, output=num_features, bias=bias, momentum=momentum),
            activation,
        )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num: int = 0, scale: int = 1):
        super().__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats: torch.Tensor, probs: torch.Tensor):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 scale: int = 1,
                 norm: str = None,
                 bias: bool = True,
                 momentum: float = 0.1,
                 ):
        super().__init__()


        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.norm = norm

        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))

        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.NormAct(norm, self.key_channels, activation, bias=bias, momentum=momentum),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.NormAct(norm, self.key_channels, activation, bias=bias, momentum=momentum),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.NormAct(norm, self.key_channels, activation, bias=bias, momentum=momentum),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.NormAct(norm, self.key_channels, activation, bias=bias, momentum=momentum),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.NormAct(norm, self.key_channels, activation, bias=bias, momentum=momentum),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.NormAct(norm, self.in_channels, activation, bias=bias, momentum=momentum),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 scale: int = 1,
                 norm: str = None,
                 bias: bool = True,
                 momentum: float = 0.1,
                 ):
        super().__init__(
            in_channels=in_channels,
            key_channels=key_channels,
            activation=activation,
            scale=scale,
            norm=norm,
            bias=bias,
            momentum=momentum,
            )


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 out_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 scale: int = 1,
                 norm: str = None,
                 bias: bool = True,
                 momentum: float = 0.1,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels=in_channels,
            key_channels=key_channels,
            activation=activation,
            scale=scale,
            norm=norm,
            bias=bias,
            momentum=momentum,
            )
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.NormAct(norm, out_channels, activation,bias=bias, momentum=momentum),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output
