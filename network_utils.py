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

def get_groups(group_mode, in_layout, out_layout) :
    """
    converts GroupModes instance (or None) into an integer
    that's suitable as an argument for the torch.conv* things
    """
#{{{
    if group_mode is None or group_mode is GroupModes.ALL :
        return 1
    elif group_mode is GroupModes.SINGLE :
        return min(in_layout.channels, out_layout.channels)
    elif group_mode is GroupModes.BUNDLE :
        return 4 if out_layout.density_channels != 0 else 3
    else :
        raise RuntimeError('invalid group mode {}'.format(group_mode))
#}}}

class Layout :
    """
    a simple wrapper around some properties we associate with tensors
    """
#{{{
    def __init__(self, channels, resolution, density_channels) :
        self.channels = channels
        self.resolution = resolution
        self.density_channels = density_channels
#}}}

class Activation(nn.Module) :
    """
    our custom activation function that works on the channels we associate with density
    and on those we associate with displacement separately
    """
#{{{
    def __init__(self, layout) :
        super().__init__()
        self.density_channels = layout.density_channels
        if self.density_channels > 0 :
            # we are free in this choice
            self.density_activation = nn.LeakyReLU()

        # here we need to keep the sign symmetry in mind
        self.displacement_activation = nn.Hardshrink()

    def forward(self, x) :
        if self.density_channels > 0 :
            x[:, :self.density_channels, ...] = self.density_activation(x[:, :self.density_channels, ...])
        x[:, self.density_channels:, ...] = self.displacement_activation(x[:, self.density_channels:, ...])
        return x
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
                 in_layout=None, out_layout=None, bias=None, group_mode=None) :
        super().__init__()
        self.resample = resample
        self.external_weight = external_weight
        
        assert (self.external_weight and in_layout is None and out_layout is None
                                     and bias is None and group_mode is None)     \
               or (not self.external_weight and in_layout is not None and out_layout is not None)

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
                          else nn.Conv3d)(in_layout.channels, out_layout.channels, 3, stride=stride,
                                          groups=get_groups(group_mode, in_layout, out_layout),
                                          bias=True if bias is None else bias)
            self.conv = lambda x, w, b, **kwargs : \
                             self.conv_(x)

    def forward(self, x, w=None, b=None, **kwargs) :
        # note that w, b, and kwargs are only used for the external_weight case
        # kwargs is intended to be used for the `groups' argument only
        assert (self.external_weight and w is not None) \
               or (not self.external_weight and w is None)

        # circular padding is not universally supported in the pytorch convolutional modules, so we'll do it manually
        if self.padding != 0 :
            x = F.pad(x, [self.padding,]*6, mode=self.padding_mode)

        # apply the convolution. Note that if not self.external_weight, the w, b arguments are ignored
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
    def __init__(self, in_layout, out_layout, N_styles,
                 bias=True, resample=Resample.EQUAL):
        super().__init__()

        in_chan = in_layout.channels
        out_chan = out_layout.channels

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
    def __init__(self, in_layout, out_layout,
                       activation=True, batch_norm=False, dropout=False, styles=False, bias=True,
                       group_mode=GroupModes.ALL, # do not change if styles=True
                       batch_norm_kw={}) :
        super().__init__()

        self.activation = (nn.Identity if not activation else Activation)(out_layout)
        self.batch_norm = (nn.Identity if not batch_norm else nn.BatchNorm3d) \
                                (**batch_norm_kw)
        self.dropout    = (nn.Identity if not dropout else nn.Dropout3d) \
                                (0.5 if isinstance(dropout, bool)
                                 else dropout if isinstance(dropout, float)
                                 else 0.5)

        resample=Resample.EQUAL if in_layout.resolution == out_layout.resolution \
                 else Resample.UP if 2*in_layout.resolution == out_layout.resolution \
                 else Resample.DOWN if in_layout.resolution == 2 * out_layout.resolution \
                 else None
        assert resample is not None, "Incompatible resolutions in=%d out=%d"%(in_layout.resolution, out_layout.resolution)
        common_conv_kwargs = dict(in_layout=in_layout, out_layout=out_layout,
                                  resample=resample, bias=bias)

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
    consistent with the in_layout, out_layout arguments.
    This module is not completely flexible, so during hyperparameter optimization the implementation
    may need to be changed.

    Rules regarding the in_layout and out_layout arguments:
        > resolutions must be equal or related by factor 2
        > if out_layout.density_channels < 0, it is inferred from the channel ratio and in_layout.density_channels
        > if settings.USE_DENSITY = False, it is still allowed to have non-zero density channels
          (this actually makes sense if we want to reduce the displacement field to some scalars)

    In the current implementation, data flow looks like this :

                                             residual [optional]
                                   ------------------------
                   resampling     |                        |
                     SINGLE       |           ALL          | [ combined using +,
                    no style      |          styles        v   divide by sqrt(2) ]
    input[in_layout] ------> [out_layout] --> ... --> output[out_layout]
    """
#{{{
    def __init__(self, in_layout, out_layout,
                       N_layers=4, residual=True,
                       activation=True, batch_norm=False, dropout=False, bias=True,
                       batch_norm_kw={}) :
        super().__init__()

        self.residual = residual

        assert(N_layers > 1)
        common_layer_kwargs = dict(activation=activation, batch_norm=batch_norm,
                                   dropout=dropout, bias=bias, batch_norm_kw=batch_norm_kw)

        if out_layout.density_channels < 0 :
            out_layout.density_channels = in_layout.density_channels * out_layout.channels // in_layout.channels
        
        layers = []
        for ii in range(N_layers) :
            layers.append(Layer(in_layout=in_layout if ii==0 else out_layout,
                                out_layout=out_layout,
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
