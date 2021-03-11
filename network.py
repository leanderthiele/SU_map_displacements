import torch
import torch.nn as nn

import settings
from network_utils import Resample, Layout, Block


class Network(nn.Module) :
    """
    the main network, currently only a debugging implementation
    """
    def __init__(self, *args, **kwargs) :
        super().__init__()
        self.block1 = Block(in_layout=Layout(4, settings.NSIDE, 1),
                            out_layout=Layout(8, settings.NSIDE//2, 2))
        self.block2 = Block(in_layout=Layout(8, settings.NSIDE//2, 2),
                            out_layout=Layout(3, settings.NSIDE, 0))

    def forward(self, x, s) :
        x = self.block1(x, s)
        x = self.block2(x, s)
        return x
