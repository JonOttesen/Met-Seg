from dataclasses import dataclass
from typing import Union, List, Tuple

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

@dataclass
class StageArgs:
    num_modules: int
    num_branches: int
    num_blocks: List[int]
    num_channels: List[int]
    block: str

hrnet_w48 = [
    StageArgs(num_modules=1, num_branches=1, num_blocks=[3], num_channels=[64], block="BOTTLENECK"),
    StageArgs(num_modules=1, num_branches=2, num_blocks=[3, 3], num_channels=[48, 96], block="BASIC"),
    StageArgs(num_modules=2, num_branches=3, num_blocks=[3, 3, 3], num_channels=[48, 96, 192], block="BASIC"),
    StageArgs(num_modules=2, num_branches=4, num_blocks=[3, 3, 3, 3], num_channels=[48, 96, 192, 384], block="BASIC"),
    ]

hrnet_w32 = [
    StageArgs(num_modules=1, num_branches=1, num_blocks=[4], num_channels=[64], block="BOTTLENECK"),
    StageArgs(num_modules=1, num_branches=2, num_blocks=[4, 4], num_channels=[32, 64], block="BASIC"),
    StageArgs(num_modules=4, num_branches=3, num_blocks=[4, 4, 4], num_channels=[32, 64, 128], block="BASIC"),
    StageArgs(num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4], num_channels=[32, 64, 128, 256], block="BASIC"),
    ]

hrnet_w18 = [
    StageArgs(num_modules=1, num_branches=1, num_blocks=[4], num_channels=[64], block="BOTTLENECK"),
    StageArgs(num_modules=1, num_branches=2, num_blocks=[4, 4], num_channels=[18, 36], block="BASIC"),
    StageArgs(num_modules=4, num_branches=3, num_blocks=[4, 4, 4], num_channels=[18, 36, 72], block="BASIC"),
    StageArgs(num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4], num_channels=[18, 36, 72, 144], block="BASIC"),
    ]
