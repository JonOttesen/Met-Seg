from typing import List

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

"""
https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
https://arxiv.org/pdf/1807.06521.pdf
"""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio: float = 1./16,
        pool_types: List[str] = ['avg', 'max'],
        **kwargs,
        ):
        super().__init__()

        self.channels = channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channels, int(channels*ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channels*ratio), channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False),
            nn.InstanceNorm3d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM3D(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio: float = 1./16,
        **kwargs,
        ):
        super().__init__()
        pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(channels=channels, ratio=ratio, pool_types=pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x: torch.Tensor):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out
