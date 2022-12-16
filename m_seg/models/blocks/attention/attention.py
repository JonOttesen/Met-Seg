from typing import Optional

import torch
import torch.nn as nn

from .cbam import SpatialGate, ChannelGate, CBAM
from .fca import FCA
from .squeeze_excitation import SqueezeExcitation

channel_methods = {
    "fca": FCA,
    "cbam": ChannelGate,
    "se": SqueezeExcitation,
}

spatial_methods = {
    "cbam": SpatialGate,
}

class Attention(nn.Module):
    CHANNEL_ATTENTIONS = ["fca", "cbam", "se"]
    SPATIAL_ATTENTIONS = ["cbam"]

    def __init__(
        self,
        channels: int,
        height: int = None,
        width: int = None,
        ratio: float = 1./8,
        channel_attention: Optional[str] = "cbam",
        spatial_attention: Optional[str] = "cbam",
        freq_sel_method: Optional[str] = 'top16',
        ):
        super().__init__()

        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

        if channel_attention is not None:
            assert channel_attention in self.CHANNEL_ATTENTIONS, "channel method {0} is not in accpected methods".format(channel_attention)
            self.channel = channel_methods[channel_attention](
                channels=channels,
                ratio=ratio,
                height=height,
                width=width,
                freq_sel_method=freq_sel_method,
                )

        if spatial_attention is not None:
            assert spatial_attention in self.SPATIAL_ATTENTIONS, "spatial method {0} is not in accpected methods".format(spatial_attention)
            self.spatial = spatial_methods[spatial_attention]()

    def forward(self, x: torch.Tensor):
        if self.channel_attention is not None:
            x = self.channel(x)
        if self.spatial_attention is not None:
            x = self.spatial(x)
        return x
