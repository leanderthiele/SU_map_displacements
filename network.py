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


                       block_in                 {                   }                block_out                 collapse
    input[in_layout] ------------> [in_layout1] {    some levels    } [out_layout1] -----------> [out_layout1] --------> output[out_layout]
                                                {                   }
                                                  -------------->
                                                   block_through

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
                              activation=Activations.NONE,
                              batch_norm=False,
                              N_layers=2, residual=False)

        # we hold the running layout in this variable as we step through the levels
        tmp_layout = deepcopy(in_layout1)

        # construct the levels
        levels = []
        for ii in range(settings.NLEVELS) :
            levels.append(Level(tmp_layout, channel_factor=4 if ii<2 else 2,
                                N_layers=settings.DEFAULT_NLAYERS))
            tmp_layout = deepcopy(levels[-1].lower_layout)
        self.levels = nn.ModuleList(levels)

        # construct the through-block at the bottom
        self.block_through = Block(tmp_layout, tmp_layout)

        if settings.RANK == 0 :
            print('Network : through-layout = %s'%str(tmp_layout))


    def forward(self, x, s, guess) :
        x = self.block_in(x, s)

        for l in self.levels :
            x = l.contract(x, s)

        x = self.block_through(x, s)

        for l in self.levels[::-1] :
            x = l.expand(x, s)

        x = self.block_out(x, s)

        # TODO maybe this is not the best way to do it!
        #      Note that guess is in [-1/2, 1/2], so it makes sense to multiply by 2 first
        #      It should be noted that the arcsin depends on the choice of the OUTPUT activation
        #      function, so this is information duplication.
        #
        #      The main problem is that after adding the `guess' to whatever network output
        #      we have, we cannot be sure anymore about the periodicity.
        #      Maybe this whole issues is actually not relevant at all for the vast majority of 
        #      particles, in which case we could just get rid of the sin etc.
        #
        #      --> TEMPORARY, CHANGE LATER
        x = self.collapse(x, s)

        return x + guess

    
    def sync_batchnorm(self) :
        # we need to convert all batch normalizations into synchronized ones!
        return nn.SyncBatchNorm.convert_sync_batchnorm(self)

    
    def to_ddp(self) :
        # turns the network into a parallel module
        return DistributedDataParallel(self, device_ids=[settings.DEVICE_IDX, ])
#}}}
