import torch
import torch.nn as nn
import torch.optim

import settings

class Loss(nn.MSELoss) :
    """
    simply change the base class (and possibly its __init__ args)
    for different functionality

    The D3M paper (which has a somewhat similar objective) uses the MSELoss.

    Note that it may be useful to include some dependency on delta_L,
    since the particles in high delta_L move a lot further than in low delta_L
    and this may cause the network to put most of its efforts towards getting the
    high delta_L right. (on the other hand, it may be argued that if it corectly
    handles high delta_L, lower values should be no problem...)
    This needs to be tested.
    """
#{{{
    def __init__(self) :
        super().__init__()
#}}}

class Optimizer(torch.optim.Adam) :
    """
    simply change the base class (and possibly its __init__ args)
    for different functionality
    """
#{{{
    def __init__(self, params) :
        super().__init__(params, **settings.OPTIMIZER_ARGS)
#}}}
