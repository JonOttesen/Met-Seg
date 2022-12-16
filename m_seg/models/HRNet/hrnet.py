# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------
# Modified and changed by Jon Andre Ottesen jon.a.ottesen@gmail.com
# Original source: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/f9fb1ba66ff8aea29d833b885f08df64e62c2b23/lib/models/seg_hrnet_ocr.py
from typing import List, Union, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

import numpy as np

from .ocr import SpatialGather_Module, SpatialOCR_Module
from .config import StageArgs, ALIGN_CORNERS

from ..blocks import (
    _norm,
    Attention,
    )

from ..UNetDecoder import UNetDecoder

from ...logger import get_logger

logger = get_logger(name=__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm: str = "batch_norm",
        bias: bool = True,
        stride: int = 1,
        channel_attention: str = "fca",
        spatial_attention: str = "cbam",
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        freq_sel_method: Optional[str] = 'top16',
        ratio: float = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        downsample: nn.Module = None,
        momentum: float = 0.1,
        ):
        super().__init__()
        self.ratio = ratio

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = _norm(norm, planes, bias, momentum=momentum)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = _norm(norm, planes, bias, momentum=momentum)
        if ratio is not None:
            self.attention = Attention(
                channels=planes * self.expansion,
                ratio=ratio,
                height=height,
                width=width,
                channel_attention=channel_attention,
                spatial_attention=spatial_attention,
                freq_sel_method=freq_sel_method,
                )

        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.ratio is not None:
            out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm: str = "batch_norm",
        bias: bool = True,
        stride: int = 1,
        channel_attention: str = "cbam",
        spatial_attention: str = "cbam",
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        freq_sel_method: Optional[str] = 'top16',
        ratio: float = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        downsample: nn.Module = None,
        momentum: float = 0.1,
        ):
        super().__init__()
        self.ratio = ratio

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = _norm(norm, planes, bias, momentum=momentum)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = _norm(norm, planes, bias, momentum=momentum)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = _norm(norm, planes * self.expansion, bias, momentum=momentum)

        if ratio is not None:
            self.attention = Attention(
                channels=planes * self.expansion,
                ratio=ratio,
                height=height,
                width=width,
                channel_attention=channel_attention,
                spatial_attention=spatial_attention,
                freq_sel_method=freq_sel_method,
                )

        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.ratio is not None:
            out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.activation(out)

        return out


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: nn.Module,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        multi_scale_output=True,
        channel_attention: str = "fca",
        spatial_attention: str = "cbam",
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        ratio: Union[float, None] = 1./8,
        norm: str = "batch_norm",
        activation: nn.Module = nn.ReLU(inplace=True),
        momentum: float = 0.1,
        bias: bool = True,
        ):

        super().__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention
        self.height = height
        self.width = width

        self.ratio = ratio
        self.norm = norm
        self.momentum = momentum
        self.bias = bias

        self.multi_scale_output = multi_scale_output

        self.activation = activation
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(
        self,
        num_branches: int,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        ):
        """
        Checks whether the inputs are correct
        """
        if num_branches != len(num_blocks):
            error_msg = 'num_branches({}) <> num_blocks({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'num_branches({}) <> num_channels({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'num_branches({}) <> num_inchannels({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index: int,
        block: nn.Module,
        num_blocks: int,
        num_channels: int,
        ):
        downsample = None
        if self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            # print(self.num_inchannels[branch_index], num_channels[branch_index], block)
            # Ensure correct shape for the first set of blocks, this is most commonly used after the bottleneck
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.num_inchannels[branch_index],
                    out_channels=num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                _norm(
                    norm=self.norm,
                    output=num_channels[branch_index] * block.expansion,
                    bias=self.bias,
                    momentum=self.momentum),
            )

        layers = []

        if int(self.num_inchannels[0]) % 16 == 0:
            freq_sel_method = "top16"
        else:
            freq_sel_method = "top" + str(self.num_inchannels[0])

        layers.append(block(
            inplanes=self.num_inchannels[branch_index],
            planes=num_channels[branch_index],
            norm=self.norm,
            bias=self.bias,
            stride=1,
            channel_attention=self.channel_attention,
            spatial_attention=self.spatial_attention,
            freq_sel_method=freq_sel_method,
            height=self.height,
            width=self.width,
            ratio=self.ratio,
            activation=self.activation,
            downsample=downsample,
            momentum=self.momentum,
            ))

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion

        for l in range(1, num_blocks[branch_index]):
            layers.append(block(
                inplanes=self.num_inchannels[branch_index],
                planes=num_channels[branch_index],
                norm=self.norm,
                bias=self.bias,
                stride=1,
                channel_attention=self.channel_attention,
                spatial_attention=self.spatial_attention,
                freq_sel_method=freq_sel_method,
                height=math.ceil(self.height/(2**(l+1))),
                width=math.ceil(self.width/(2**(l+1))),
                ratio=self.ratio,
                activation=self.activation,
                downsample=None,  # Check if this is supposed to be None
                momentum=self.momentum,
                ))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _fuse_layer(self, start, end):
        """
        Args:
            start (int): the starting index to sample to
            end (int): the end index the fusing ends at
            The zero'eth index represents the highest resolution
        """
        num_inchannels = self.num_inchannels
        if start == end:
            return nn.Identity()
        elif start > end:  # Upsampling
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=num_inchannels[start],
                    out_channels=num_inchannels[end],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                _norm(
                    norm=self.norm, output=num_inchannels[end],
                    bias=self.bias, momentum=self.momentum),
                )
        else:
            down_layers = list()
            # Loop from the starting resolution down to the second to bottom resolution, i.e, end - 1
            for _ in range(end - start - 1):
                down_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=num_inchannels[start],
                        out_channels=num_inchannels[start],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False),
                    _norm(
                        norm=self.norm, output=num_inchannels[start],
                        bias=self.bias, momentum=self.momentum),
                    self.activation,
                    ))
            down_layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=num_inchannels[start],
                    out_channels=num_inchannels[end],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                _norm(
                    norm=self.norm, output=num_inchannels[end],
                    bias=self.bias, momentum=self.momentum),
                ))
            return nn.Sequential(*down_layers)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        fuse_layers = []
        for end in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for start in range(num_branches):
                fuse_layer.append(self._fuse_layer(start=start, end=end))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x: Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]):

        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        branches = list()
        for i in range(self.num_branches):
            branches.append(self.branches[i](x[i]))

        fused = list()
        # Comment: The interpolation method work since start_layer is always
        # the up-top most layer from the previous layers
        for end, end_layers in enumerate(self.fuse_layers):
            for start, start_layer in enumerate(end_layers):
                if start == 0:
                    y = start_layer(branches[start])
                else:
                    out = start_layer(branches[start])
                    if start > end:  #  If we have to upsample
                        out = F.interpolate(out, size=[y.shape[-2], y.shape[-1]], mode='bilinear', align_corners=ALIGN_CORNERS)
                    y = y + out
            
            fused.append(self.activation(y))

        return fused

class HighResolutionNet(nn.Module):

    def __init__(
        self,
        config: List[StageArgs],
        inp_classes: int,
        num_classes: int,
        ratio: Union[float, None] = 1./8,
        channel_attention: str = "cbam",
        spatial_attention: str = "cbam",
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        norm: str = "batch_norm",
        activation: nn.Module = nn.ReLU(inplace=True),
        momentum: float = 0.1,
        bias: bool = True,
        multi_scale_output: bool = True,
        ocr_mid_channels: int = 512,
        ocr_key_channels: int = 256,
        ocr_dropout: float = 0.05,
        ocr: bool = False,
        mscale: bool = False,
        scale_factor: float = 2,
        deep_supervision: bool = True,
        ):
        super().__init__()

        self.ratio = ratio
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention
        self.height = height
        self.width = width
        self.scale_factor = scale_factor
        self.deep_supervision = deep_supervision

        self.norm = norm
        self.activation = activation
        self.momentum = momentum
        self.bias = bias
        self.multi_scale_output = multi_scale_output
        self.mscale = mscale
        self.ocr = ocr

        # stem net
        self.stem = nn.Sequential(
            nn.Conv2d(inp_classes, 64, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(norm=norm, output=64, bias=bias, momentum=momentum),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, stride=int(self.scale_factor), padding=1, bias=False),
            _norm(norm=norm, output=64, bias=bias, momentum=momentum),
            activation,
            )
        
        # The HR-Net part

        num_inchannels = [64]
        stages = list()
        transitions = list()
        deep_supervision_layers = list()

        for i, stage in enumerate(config):
            # Make HRModule
            new_stage, num_inchannels = self._make_stage(
                layer_config=stage,
                num_inchannels=num_inchannels,
                multi_scale_output=True if i < len(config) - 1 else multi_scale_output,
                )

            stages.append(new_stage)
            if i < len(config) - 1:
                deep_supervision_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=num_inchannels[0],
                        out_channels=num_inchannels[0],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False),
                    _norm(
                        norm=self.norm, output=num_inchannels[0],
                        bias=self.bias, momentum=self.momentum),
                    self.activation,
                    nn.Conv2d(
                        in_channels=num_inchannels[0],
                        out_channels=num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                        )))

            # Transition to more resolution layers
            if i < len(config) - 1:
                if stage.num_branches < config[i+1].num_branches:
                    next_block = blocks_dict[config[i+1].block]
                    pre_stage_channels = [
                        channels * next_block.expansion for channels in config[i+1].num_channels]

                    transitions.append(self._make_transition_layer(num_inchannels, pre_stage_channels))
                    num_inchannels = pre_stage_channels


        self.stages = nn.ModuleList(stages)
        self.transitions = nn.ModuleList(transitions)
    
        # Always make them since they are so small it doesn't really matter
        self.deep_supervision_layers = nn.ModuleList(deep_supervision_layers)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        if self.ocr:
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(
                    in_channels=last_inp_channels,
                    out_channels=ocr_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                _norm(
                    norm=self.norm, output=ocr_mid_channels,
                    bias=self.bias, momentum=self.momentum),
                self.activation,
                )
            self.ocr_gather_head = SpatialGather_Module(num_classes)

            self.ocr_distri_head = SpatialOCR_Module(
                in_channels=ocr_mid_channels,
                key_channels=ocr_key_channels,
                out_channels=ocr_mid_channels,
                activation=self.activation,
                scale=1,
                norm=norm,
                bias=bias,
                momentum=momentum,
                dropout=ocr_dropout,
                )

            self.cls_head = nn.Conv2d(
                ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            _norm(
                norm=self.norm, output=last_inp_channels,
                bias=self.bias, momentum=self.momentum),
            self.activation,
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                )
            )

    def _make_transition_layer(
        self,
        num_channels_pre_layer: List[int],
        num_channels_cur_layer: List[int],
        ):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels=num_channels_pre_layer[i],
                                  out_channels=num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        _norm(
                            norm=self.norm, output=num_channels_cur_layer[i],
                            bias=self.bias, momentum=self.momentum),
                        self.activation))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=inchannels,
                            out_channels=outchannels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
                        _norm(
                            norm=self.norm, output=outchannels,
                            bias=self.bias, momentum=self.momentum),
                        self.activation))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config: StageArgs, num_inchannels: int, multi_scale_output=True):
        num_modules = layer_config.num_modules
        num_branches = layer_config.num_branches
        num_blocks = layer_config.num_blocks
        num_channels = layer_config.num_channels
        block = blocks_dict[layer_config.block]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    block=block,
                    num_blocks=num_blocks,
                    num_inchannels=num_inchannels,
                    num_channels=num_channels,
                    multi_scale_output=multi_scale_output,
                    channel_attention=self.channel_attention,
                    spatial_attention=self.spatial_attention,
                    height=self.height,
                    width=self.width,
                    ratio=self.ratio,
                    norm=self.norm,
                    activation=self.activation,
                    momentum=self.momentum,
                    bias=self.bias,
                    ))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x0_h, x0_w = x[0].size(-2), x[0].size(-1)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')

        x = self.stem(x)

        if self.deep_supervision:
            auxiliary = list()

        x = [x]

        for i in range(len(self.stages) - 1):
            x = self.stages[i](x)
            if self.deep_supervision:
                auxiliary.append(self.deep_supervision_layers[i](x[0]))
            transitioned = list()
            for j, transition in enumerate(self.transitions[i]):
                # If length of transitions are larger than the number of previous resolutions
                if j < len(self.transitions[i]) - 1:
                    transitioned.append(transition(x[j]))
                else:
                    transitioned.append(transition(x[-1]))
            x = transitioned

        x = self.stages[-1](x)


        # Upsampling
        # x0_h, x0_w = x[0].size(-2), x[0].size(-1)
        x0 = F.interpolate(x[0], size=(int(x0_h), int(x0_w)),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x1 = F.interpolate(x[1], size=(int(x0_h), int(x0_w)),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(int(x0_h), int(x0_w)),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(int(x0_h), int(x0_w)),
                        mode='bilinear', align_corners=ALIGN_CORNERS)

        # feats = torch.cat([x[0], x1, x2, x3], 1)
        feats = torch.cat([x0, x1, x2, x3], 1)

        # ocr
        out_aux = self.aux_head(feats)
        
        if not self.ocr:
            if self.mscale:
                if self.deep_supervision:
                    return out_aux, auxiliary, feats
                else:
                    return out_aux, feats
            if self.deep_supervision:
                return out_aux, auxiliary
            else:
                return out_aux

        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)

        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        if self.mscale:
            return out, auxiliary, out_aux, feats

        return out, auxiliary
