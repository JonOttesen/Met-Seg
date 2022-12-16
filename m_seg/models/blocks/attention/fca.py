from typing import Optional, List

import math
import torch
import torch.nn as nn

"""
https://github.com/cfzd/FcaNet/blob/master/model/fcanet.py
https://arxiv.org/pdf/2012.11879.pdf
"""

def get_freq_indices(method):
    # This is originally in the original code Jon A
    # assert method in ['top1','top2','top4','top8','top16','top32',
                      # 'bot1','bot2','bot4','bot8','bot16','bot32',
                      # 'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class FCA(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        ratio: Optional[float] = 1./16,
        freq_sel_method: Optional[str] = 'top16',
        ):
        super().__init__()
        self.ratio = ratio
        self.height = height
        self.width = width

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (height // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (width // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(height, width, mapper_x, mapper_y, channels)

        self.fc = nn.Sequential(
            nn.Linear(channels, int(channels*ratio), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(int(channels*ratio), channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.height or w != self.width:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.height, self.width))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(
        self,
        height: int,
        width: int,
        mapper_x: List[int],
        mapper_y: List[int],
        channels: int,
        ):
        super().__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channels % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channels))


    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channels):
        dct_filter = torch.zeros(channels, tile_size_x, tile_size_y)

        c_part = channels // len(mapper_x)
        counter = 0

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    counter += 1

        return dct_filter
