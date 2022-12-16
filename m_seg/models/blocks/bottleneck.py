# External standard modules
from typing import Optional, Callable, Union

# External third party modules
import torch
import torch.nn as nn

# Internal modules
from .squeeze_excitation import SqueezeExcitation

class Bottleneck(nn.Module):
    """
    Original paper:
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    Inspiration from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 channels: int,
                 mid_channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 ratio: float = 1./16,
                 norm: str = 'batch_norm',
                 activation: Union[Callable[..., nn.Module], None] = nn.ReLU(inplace=True),
                 downsample: Optional[nn.Module] = None,
                 ):
        super().__init__()

        self.stride = stride
        if stride > 1 and downsample is None:
            downsample = nn.Conv2d(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                )

        self.downsample = downsample


        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            )
        self.norm1 = self._norm(norm, mid_channels, bias=bias)

        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            groups=groups,
            bias=False,
            padding=1,
            )
        self.norm2 = self._norm(norm, mid_channels, bias=bias)

        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=False
            )
        self.norm3 = self._norm(norm, channels, bias=bias)

        self.activation = activation

        self.se = SqueezeExcitation(channels=channels, ratio=ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.se(x)

        if self.stride > 1:
            identity = self.downsample(identity)

        x += identity
        x = self.activation(x)

        return x

    def _norm(self, norm: str, output: int, bias: bool):
        if norm == "batch_norm":
            return nn.BatchNorm2d(
                num_features=output,
                affine=bias
                )
        elif norm == "instance_norm":
            return nn.InstanceNorm2d(
                num_features=output,
                affine=bias
                )
        else:
            return nn.LayerNorm(
                normalized_shape=1,
                elementwise_affine=bias,
                )



class BottleneckV2(nn.Module):
    """
    Original paper:
    https://arxiv.org/pdf/1603.05027.pdf
    Inspiration from:
    https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/resnet.py#L135
    """

    def __init__(self,
                 channels: int,
                 mid_channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 ratio: float = 1./16,
                 norm: str = 'batch_norm',
                 activation: Union[Callable[..., nn.Module], None] = nn.ReLU(inplace=True),
                 downsample: Optional[nn.Module] = None,
                 ):
        super().__init__()

        self.stride = stride
        if stride > 1 and downsample is None:
            downsample = nn.Conv2d(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                )

        self.downsample = downsample

        self.norm0 = self._norm(norm, channels, bias=bias)

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            )
        self.norm1 = self._norm(norm, mid_channels, bias=bias)

        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            groups=groups,
            bias=False,
            padding=1,
            )
        self.norm2 = self._norm(norm, mid_channels, bias=bias)

        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=False
            )

        self.activation = activation


        self.se = SqueezeExcitation(channels=channels, ratio=ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm0(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)

        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv3(x)

        x = self.se(x)

        if self.stride > 1:
            identity = self.downsample(identity)

        x += identity

        return x

    def _norm(self, norm: str, output: int, bias: bool):
        if norm == "batch_norm":
            return nn.BatchNorm2d(
                num_features=output,
                affine=bias
                )
        elif norm == "instance_norm":
            return nn.InstanceNorm2d(
                num_features=output,
                affine=bias
                )
        else:
            return nn.LayerNorm(
                normalized_shape=1,
                elementwise_affine=bias,
                )

