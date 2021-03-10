import math
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F

import settings

class Resample(Enum) :
    """
    resampling modes -- UP / DOWN are factor 2 resamplings (stride=2)
    """
    UP = auto() # upsampling -- resolution increases by 2
    DOWN = auto() # downsampling -- resolution decreases by 2
    EQUAL = auto() # resolution stays the same

class GroupModes(Enum) :
    """
    how we want to group the channels in convolutions
    """
    ALL = auto() # all channels are convolved with one another
    SINGLE = auto() # each channel is treated individually
    BUNDLE = auto() # each dimension is treated separately.
                    # assumes that each direction corresponds to the same number of channels
                    # and the density field, if present, also has this number of channels

def crop(x, num=1) :
    """
    removes num rows on each side in each dimension of x
    x is assumed to have shape [batch, channel, x1, x2, x3]
    """
#{{{
    return x.narrow(2, num/2, x.shape[2]-num) \
            .narrow(3, num/2, x.shape[3]-num) \
            .narrow(4, num/2, x.shape[4]-num) \
            .contiguous()
#}}}

def get_groups(group_mode, in_chan, out_chan) :
    """
    converts GroupModes instance (or None) into an integer
    that's suitable as an argument for the torch.conv* things
    """
#{{{
    if group_mode is None or group_mode is GroupModes.ALL :
        return 1
    elif group_mode is GroupModes.SINGLE :
        return min(in_chan, out_chan)
    elif group_mode is GroupModes.BUNDLE :
        return 4 if settings.USE_DENSITY else 3
    else :
        raise RuntimeError('invalid group mode {}'.format(group_mode))
#}}}

class Conv3d(nn.Module) :
    """
    implements 3d convolution with periodic boundary conditions,
    consistently treating the cropping and padding
    It has two modes, depending on the external_weight keyword.
    We expect that external_weight=False for all applications except within StyleConv3d.
    """
#{{{
    def __init__(self, resample=Resample.EQUAL, external_weight=False,
                 # following arguments only to be used if external_weight=False,
                 # otherwise an exeception will be raised
                 in_chan=None, out_chan=None, bias=None, group_mode=None) :
        super().__init__()
        self.resample = resample
        self.external_weight = external_weight
        
        assert (self.external_weight and in_chan is None and out_chan is None
                                     and bias is None and group_mode is None)     \
               or (not self.external_weight and in_chan is not None and out_chan is not None)

        stride = 1 if self.resample is Resample.EQUAL else 2

        self.padding = 0 if self.resample is Resample.UP else 1
        self.padding_mode = 'circular' # periodic boundary conditions

        if group_mode is not None :
            assert not external_weight

        if external_weight :
            self.conv = lambda x, w, b, **kwargs : \
                            (F.conv_transpose3d
                             if self.resample is Resample.UP
                             else F.conv3d)(x, w, bias=b, stride=stride, **kwargs)
        else :
            self.conv_ = (nn.ConvTranspose3d
                          if self.resample is Resample.UP
                          else nn.Conv3d)(in_chan, out_chan, 3, stride=stride,
                                          padding=self.padding, padding_mode=self.padding_mode,
                                          groups=get_groups(group_mode, in_chan, out_chan),
                                          bias=True if bias is None else bias)
            self.conv = lambda x, w, b, **kwargs : \
                             self.conv_(x)

    def forward(self, x, w=None, b=None, **kwargs) :
        # note that w, b, and kwargs are only used for the external_weight case
        # kwargs is intended to be used for the `groups' argument only
        assert (self.external_weight and w is not None) \
               or (not self.external_weight and w is None)

        # the functional.conv3d doesn't seem to support circular padding,
        # so we'll do this ourselves
        if self.external_weight and self.padding :
            x = F.pad(x, [self.padding,]*6, mode=self.padding_mode)

        # apply the convolution. Note that if not self.external_weight, the w argument is ignored
        x = self.conv(x, w, b, **kwargs)

        # if we did upsampling, we'll need to remove one row on each side in each dimension
        if self.resample is Resample.UP :
            x = crop(x)

        return x
#}}}

class StyleConv3d(nn.Module) :
    """
    Convolution layer with modulation and demodulation, from StyleGAN2.
    Weight and bias initialization from `torch.nn._ConvNd.reset_parameters()`.
    This module convolves all channels with one other.
    It implements the forward method as
        forward(x, s)
    where x is the input tensor [Nbatches, Nchannels, dim, dim, dim]
      and s is the style tensor [Nbatches, N_styles]
    """
#{{{
    def __init__(self, in_chan, out_chan, N_styles,
                 bias=True, resample=Resample.EQUAL):
        super().__init__()

        self.resample = resample
        if self.resample is Resample.EQUAL :
            K3 = (3,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
        elif self.resample is Resample.UP :
            K3 = (2,) * 3
            # NOTE not clear to me why convtranspose have channels swapped
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
        elif self.resample is Resample.DOWN :
            K3 = (2,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))

        self.conv = Conv3d(resample=resample, external_weight=True)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5),
                                 mode='fan_in',  # effectively 'fan_out' for 'D'
                                 nonlinearity='leaky_relu')

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.style_weight = nn.Parameter(torch.empty(in_chan, N_styles))
        nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='leaky_relu')
        self.style_bias = nn.Parameter(torch.ones(in_chan))  # NOTE: init to 1

    def forward(self, x, s, eps=1e-8) :
        N, Cin, *DHWin = x.shape
        C0, C1, *K3 = self.weight.shape
        if self.resample is Resample.UP :
            Cin, Cout = C0, C1
        else:
            Cout, Cin = C0, C1

        s = F.linear(s, self.style_weight, bias=self.style_bias)

        # modulation
        if self.resample is Resample.UP :
            s = s.reshape(N, Cin, 1, 1, 1, 1)
        else:
            s = s.reshape(N, 1, Cin, 1, 1, 1)
        w = self.weight * s

        # demodulation
        if self.resample is Resample.UP :
            fan_in_dim = (1, 3, 4, 5)
        else:
            fan_in_dim = (2, 3, 4, 5)
        w *= torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        # LFT : this piece of code is a bit tricky -- what it does is to have a different weight
        #       for each batch, still performing this operation in a single call
        w = w.reshape(N * C0, C1, *K3)
        x = x.reshape(1, N * Cin, *DHWin)
        x = self.conv(x, w, self.bias, groups=N)
        _, _, *DHWout = x.shape
        x = x.reshape(N, Cout, *DHWout)

        return x
#}}}

class Layer(nn.Module) :
    """
    a single convolutional layer, including activation, batch norm, dropout
    """
#{{{
    def __init__(self, in_chan, out_chan,
                       resample=Resample.EQUAL, activation=nn.LeakyReLU,
                       batch_norm=False, dropout=False, styles=False, bias=True,
                       group_mode=GroupModes.ALL, # do not change if styles=True
                       activation_kw={}, batch_norm_kw={}) :
        super().__init__()

        self.activation = (nn.Identity if activation is None else activation) \
                                (**activation_kw)
        self.batch_norm = (nn.Identity if not batch_norm else nn.BatchNorm3d) \
                                (**batch_norm_kw)
        self.dropout    = (nn.Identity if not dropout else nn.Dropout3d) \
                                (0.5 if isinstance(dropout, bool)
                                 else dropout if isinstance(dropout, float)
                                 else **dropout if isinstance(dropout, dict))

        common_conv_kwargs = dict(in_chan=in_chan, out_chan=out_chan, resample=resample, bias=bias)

        if styles :
            assert isinstance(styles, int)
            assert group_mode is GroupModes.ALL
            self.do_styles = True
            self.conv = StyleConv3d(N_styles=styles, **common_conv_kwargs)
        else :
            self.do_styles = False
            self.conv = Conv3d(external_weight=False, group_mode=group_mode, **common_conv_kwargs)

    def forward(self, x, s=None) :
        if self.do_styles :
            return self.activation(self.batch_norm(self.dropout(self.conv(x, s))))
        else :
            return self.activation(self.batch_norm(self.dropout(self.conv(x))))
#}}}

class Block(nn.Module) :
    """
    a linear stack of layers, possibly with a residual connection
    We guarantee that the aggregate action of the Block leads to a change in tensor shape
    consistent with the resample, in_chan, out_chan arguments.
    This module is not completely flexible, so during hyperparameter optimization the implementation
    may need to be changed
    In the current implementation, data flow looks like this :

                                             residual
                                   ------------------------
                   resampling     |                        |
                     SINGLE       |           ALL          | [ combined using +,
                    no style      |          styles        v   divide by sqrt(2) ]
    input[in_chan] ---------> [out_chan] ---> ... ---> [out_chan] ---> return
    """
#{{{
    def __init__(self, resample, in_chan, out_chan, N_layers=4, residual=False,
                       activation=nn.LeakyReLU, batch_norm=False, dropout=False, bias=True,
                       activation_kw={}, batch_norm_kw={}) :
        super().__init__()

        self.residual = residual

        assert(N_layers > 1)
        common_layer_kwargs = dict(activation=activation, batch_norm=batch_norm,
                                   dropout=dropout, bias=bias,
                                   activation_kw=activation_kw, batch_norm_kw=batch_norm_kw)
        
        layers = []
        for ii in range(N_layers) :
            layers.append(Layer(in_chan=in_chan if ii==0 else out_chan,
                                out_chan=out_chan,
                                resample=resample if ii==0 else Resample.EQUAL,
                                styles=False if ii==0 else settings.NSTYLES,
                                group_mode=GroupModes.SINGLE if ii==0 else GroupModes.ALL,
                                **common_layer_kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, s) :
        x = self.layers[0](x, s)
        
        if self.residual :
            xres = torch.clone(x)

        for _, l in enumerate(self.layers, start=1) :
            x = l(x, s)

        if self.residual :
            x += x0
            x /= math.sqrt(2)

        return x
#}}}
