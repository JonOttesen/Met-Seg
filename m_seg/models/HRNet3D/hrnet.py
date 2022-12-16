# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------
# Modified and changed by Jon Andre Ottesen jon.a.ottesen@gmail.com
# Original source: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/f9fb1ba66ff8aea29d833b885f08df64e62c2b23/lib/models/seg_hrnet_ocr.py
from turtle import forward
from typing import List, Union, Optional
import math
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

import numpy as np

from .config import StageArgs, ALIGN_CORNERS, hrnet_w18, hrnet_w32
from .cbam3d import CBAM3D

from ...logger import get_logger

logger = get_logger(name=__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        bias: bool = True,
        stride: int = 1,
        ratio: float = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        downsample: nn.Module = None,
        ):
        super().__init__()
        self.ratio = ratio

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = nn.InstanceNorm3d(planes, affine=bias)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = nn.InstanceNorm3d(planes, affine=bias)
        if self.ratio is not None:
            self.attention = CBAM3D(
                channels=planes*self.expansion,
                ratio=ratio,
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
        bias: bool = True,
        stride: int = 1,
        ratio: float = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        downsample: nn.Module = None,
        ):
        super().__init__()
        self.ratio = ratio

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(planes, affine=bias)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(planes, affine=bias)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.InstanceNorm3d(planes * self.expansion, affine=bias)
        if self.ratio is not None:
            self.attention = CBAM3D(
                channels=planes*self.expansion,
                ratio=ratio,
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

class UNetDecoder(nn.Module):
    def __init__(
        self,
        num_channels: List[int],
        activation: nn.Module = nn.ReLU(inplace=True),
        bias: bool = True,
        ):
        super().__init__()
        self.bias = bias
        self.activation = activation
        up_convs = list()
        double_convs = list()

        for i in range(1, len(num_channels)):
            up_convs.append(self._make_upscale_conv(in_channels=num_channels[-i], out_channels=num_channels[-i-1]))
            double_convs.append(self._make_double_conv(channels=num_channels[-i-1]))

        self.up_convs = nn.ModuleList(up_convs)
        self.double_convs = nn.ModuleList(double_convs)

    def _make_upscale_conv(self, in_channels: int, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(num_features=out_channels, affine=self.bias),
            self.activation,
        )

    def _make_double_conv(self, channels: int):
        return nn.Sequential(
            nn.Conv3d(in_channels=int(2*channels), out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=channels, affine=self.bias),
            self.activation,
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=channels, affine=self.bias),
            self.activation,
        )

    def forward(self, x: List[torch.Tensor]):

        x0 = F.interpolate(x[-1], size=(x[-2].shape[-3], x[-2].shape[-2], x[-2].shape[-1]),
                        mode='trilinear', align_corners=ALIGN_CORNERS)

        x0 = self.up_convs[0](x0)

        for i in range(1, len(x)):
            x0 = self.double_convs[i-1](torch.cat([x0, x[-i-1]], dim=1))
            if i < len(x) - 1:
                shape = x[-i-2].shape
                x0 = F.interpolate(x[-i - 1], size=(shape[-3], shape[-2], shape[-1]),
                    mode='trilinear', align_corners=ALIGN_CORNERS)
                x0 = self.up_convs[i](x0)
        return x0

class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: nn.Module,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        multi_scale_output=True,
        ratio: Union[float, None] = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        bias: bool = True,
        ):

        super().__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.ratio = ratio
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
                nn.Conv3d(
                    in_channels=self.num_inchannels[branch_index],
                    out_channels=num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                nn.InstanceNorm3d(
                    num_features=num_channels[branch_index] * block.expansion,
                    affine=self.bias,
                    ),
            )

        layers = []


        layers.append(block(
            inplanes=self.num_inchannels[branch_index],
            planes=num_channels[branch_index],
            bias=self.bias,
            stride=1,
            ratio=self.ratio,
            activation=self.activation,
            downsample=downsample,
            ))

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion

        for l in range(1, num_blocks[branch_index]):
            layers.append(block(
                inplanes=self.num_inchannels[branch_index],
                planes=num_channels[branch_index],
                bias=self.bias,
                stride=1,
                ratio=self.ratio,
                activation=self.activation,
                downsample=None,  # Check if this is supposed to be None
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
                nn.Conv3d(
                    in_channels=num_inchannels[start],
                    out_channels=num_inchannels[end],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.InstanceNorm3d(
                    num_features=num_inchannels[end],
                    affine=self.bias)
                )
        else:
            down_layers = list()
            # Loop from the starting resolution down to the second to bottom resolution, i.e, end - 1
            for _ in range(end - start - 1):
                down_layers.append(nn.Sequential(
                    nn.Conv3d(
                        in_channels=num_inchannels[start],
                        out_channels=num_inchannels[start],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False),
                    nn.InstanceNorm3d(
                        num_features=num_inchannels[start],
                        affine=self.bias),
                    self.activation,
                    ))
            down_layers.append(nn.Sequential(
                nn.Conv3d(
                    in_channels=num_inchannels[start],
                    out_channels=num_inchannels[end],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                nn.InstanceNorm3d(
                    num_features=num_inchannels[end],
                    affine=self.bias),
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

    def forward(self, x: List[torch.Tensor]):
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
                        out = F.interpolate(out, size=[y.shape[-3], y.shape[-2], y.shape[-1]], mode='trilinear', align_corners=ALIGN_CORNERS)
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
        activation: nn.Module = nn.ReLU(inplace=True),
        bias: bool = True,
        multi_scale_output: bool = True,
        deep_supervision: bool = True,
        ):
        super().__init__()

        self.ratio = ratio

        self.activation = activation
        self.bias = bias
        self.multi_scale_output = multi_scale_output
        self.deep_supervision = deep_supervision

        num_inchannels = [32]
        start_channels = num_inchannels
        self.stem = nn.Sequential(
            nn.Conv3d(inp_classes, num_inchannels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=num_inchannels[0], affine=bias),
            self.activation,
            )

        # """
        # The HR-Net part

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

            # Deep supervision layers
            if i < len(config) - 1:
                deep_supervision_layers.append(
                    nn.Conv3d(
                        in_channels=num_inchannels[0],
                        out_channels=num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                        ))

            # Transition to more resolution layers
            if i < len(config) - 1:
                if stage.num_branches < config[i+1].num_branches:
                    next_block = blocks_dict[config[i+1].block]
                    next_stage_channels = [
                        channels * next_block.expansion for channels in config[i+1].num_channels]

                    transitions.append(self._make_transition_layer(num_inchannels, next_stage_channels))
                    num_inchannels = next_stage_channels


        self.stages = nn.ModuleList(stages)
        self.transitions = nn.ModuleList(transitions)
        if self.deep_supervision:
            self.deep_supervision_layers = nn.ModuleList(deep_supervision_layers)


        self.decoder = UNetDecoder(next_stage_channels, bias=self.bias, activation=self.activation)

        self.output = nn.Conv3d(
                in_channels=next_stage_channels[0],
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
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
                        nn.Conv3d(in_channels=num_channels_pre_layer[i],
                                  out_channels=num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.InstanceNorm3d(
                            num_features=num_channels_cur_layer[i],
                            affine=self.bias),
                        self.activation))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv3d(
                            in_channels=inchannels,
                            out_channels=outchannels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
                        nn.InstanceNorm3d(
                            num_features=outchannels,
                            affine=self.bias),
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
                    ratio=self.ratio,
                    activation=self.activation,
                    bias=self.bias,
                    ))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x0_h, x0_w, x0_d = x.size(-3), x.size(-2), x.size(-1)

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

        x = self.decoder(x)

        out = self.output(x)

        if self.deep_supervision:
            return out, auxiliary
        return out
        # Previous non-UNet like decoder

        # Divide by 2: 16761MiB
        # Not divide by 2: 33379MiB
        # UNet like decoder: 18095MiB
        x3 = F.interpolate(x[3], size=(x0_h, x0_w, x0_d),
            mode='trilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)
        x0 = F.interpolate(x[0], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x0, x1, x2, x3], 1)


        feats = self.combine(feats)
        # feats = F.interpolate(feats, size=(x0_h, x0_w, x0_d),
                        # mode='trilinear', align_corners=ALIGN_CORNERS)
        # Memory: 4411 if stopped here
        out = self.output(self.activation(feats + x_stem))

        # Memory: 5467 is stopped here
        if self.deep_supervision:
            return out, auxiliary
        return out

if __name__=='__main__':
    import time
    model = HighResolutionNet(
            config=hrnet_w18,
            inp_classes=4,
            num_classes=1,
            ratio=1./8,
            activation=nn.ReLU(inplace=True),
            bias=True,
            multi_scale_output=True,
            scale_factor=2,
            )

    model = model.to('cuda:0')
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('The number of params in Million: ', params/1e6)
    a = torch.rand((1, 4, 48, 48, 48)).to('cuda:0')
    start = time.time()
    b = model(a)
    print(b.shape)
    print(time.time() - start)
    time.sleep(100)