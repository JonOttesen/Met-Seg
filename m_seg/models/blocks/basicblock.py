# External standard modules
from typing import Optional, Callable, Union

# External third party modules
import torch
import torch.nn as nn

# Internal modules
from .attention import Attention
from .norm import _norm

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm: str = "batch_norm",
        bias: bool = True,
        ratio: float = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        momentum: float = 0.1,
        ):
        super().__init__()
        self.ratio = ratio
        stride = 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = _norm(norm, planes, bias, momentum=momentum)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = _norm(norm, planes, bias, momentum=momentum)
        if inplanes != planes:
            self.conv3 = conv3x3(inplanes, planes, stride)
            self.norm3 = _norm(norm, planes, bias, momentum=momentum)
        else:
            self.conv3 = None
            self.norm3 = None

        if ratio is not None:
            self.attention = Attention(
                channels=planes,
                ratio=ratio,
                channel_attention="cbam",
                spatial_attention="cbam",
                )

        self.activation = activation
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

        if self.conv3 is not None and self.norm3 is not None:
            residual = self.conv3(residual)
            residual = self.norm3(residual)

        out = out + residual
        out = self.activation(out)

        return out