from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import settings
from network_utils import Activations, Layout, Block, Level


class Network(nn.Module) :
    """
    the main network.
    We use a standard UNet architecture, looking like


                       block_in                 {                   }                block_out               collapse
    input[in_layout] ------------> [in_layout1] {    some levels    } [out_layout1] -----------> out_layout1 --------> output[out_layout]
                                                {                   }
                                                  -------------->
                                                   block through

    We may want to do something special in the initial and final blocks,
    so we keep them separate from the relatively rigid level structure.

    Due to the current limitations, out_layout1 = in_layout1.
    """
#{{{
    def __init__(self, *args, **kwargs) :
        super().__init__()

        in_layout = Layout(channels=4 if settings.USE_DENSITY else 3,
                           resolution=settings.NSIDE,
                           density_channels=1 if settings.USE_DENSITY else 0)
        out_layout = Layout(channels=3,
                            resolution=settings.NSIDE,
                            density_channels=0)

        block_in_channel_factor = 2
        in_layout1 = Layout(channels=block_in_channel_factor*in_layout.channels,
                            resolution=in_layout.resolution,
                            density_channels=block_in_channel_factor*in_layout.density_channels)

        # construct the special blocks
        self.block_in = Block(in_layout, in_layout1)

        self.block_out = Block(in_layout1, in_layout1)

        # in the output, we don't want any activation function because we want to
        # map to the entire real line
        self.collapse = Block(in_layout1, out_layout,
                              activation=Activations.OUTPUT,
                              batch_norm=False,
                              N_layers=2, residual=False)

        # we hold the running layout in this variable as we step through the levels
        tmp_layout = deepcopy(in_layout1)

        # construct the levels
        levels = []
        Nlevels = 5 # note : this is the maximum we can do on our 16 GB cards
        for ii in range(Nlevels) :
            levels.append(Level(tmp_layout, channel_factor=4 if ii<2 else 2))
            tmp_layout = deepcopy(levels[-1].lower_layout)
        self.levels = nn.ModuleList(levels)

        # construct the through-block at the bottom
        self.block_through = Block(tmp_layout, tmp_layout)

        print('Network : through-layout = %s'%str(tmp_layout))


    def forward(self, x, s) :
        x = self.block_in(x, s)

        for l in self.levels :
            x = l.contract(x, s)

        x = self.block_through(x, s)

        for l in self.levels[::-1] :
            x = l.expand(x, s)

        x = self.block_out(x, s)
        x = self.collapse(x, s)

        return x

    
    def to_ddp(self, rank, world_size) :
        # turns the network into a parallel module

        # we need to convert all batch normalizations into synchronized ones!
        return DistributedDataParallel(nn.SyncBatchNorm.convert_sync_batchnorm(self),
                                       device_ids=[rank])
#}}}
