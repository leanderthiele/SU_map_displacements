"""
Defines the Network class, which represents our neural net.
"""

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


    def forward(self, x, s, g) :
        """
        x = input field (normalized)
        s = style vector (normalized)
        g = our naive guess for what the output should be
            (currently this is just x but with a different normalization)
        """
        
        x = self.block_in(x, s)

        for l in self.levels :
            x = l.contract(x, s)

        x = self.block_through(x, s)

        for l in self.levels[::-1] :
            x = l.expand(x, s)

        x = self.block_out(x, s)

        x = self.collapse(x, s)

        # map to [-0.5, 0.5] to avoid exploding loss
        # FIXME somehow all these periodic outputs don't really work yet --
        #       usually training is ok for like 40 epochs and then the loss explodes
        #       and remains almost constant
        #       --> seems to indicate that we get stuck somewhere,
        #           maybe we should try a different method to `periodicize'
        #return Network.sawteeth(x + guess)
        return x + g
    
    
    @staticmethod
    def sawteeth(x) :
        """
        helper function.
        x is centered on zero and goes from -inf to inf
        returns x centered on zero going from -1/2 to 1/2
        """
        return torch.remainder(x+0.5, 1.0) - 0.5

    
    def sync_batchnorm(self) :
        """
        we need to convert all batch normalizations into synchronized ones!

        NOTE this method is not used at the moment because we believe that batch normalization
             does not play well with the style modulation.
        """

        return nn.SyncBatchNorm.convert_sync_batchnorm(self)

    
    def to_ddp(self) :
        """
        turns the network into a parallel module
        """

        return DistributedDataParallel(self, device_ids=[settings.DEVICE_IDX, ])
#}}}
