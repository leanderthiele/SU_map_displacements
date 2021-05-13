import math
from enum import Enum, auto
from copy import deepcopy

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

class Activations(Enum) :
    """
    which activation function to apply
    """
    STANDARD = auto() # for the StandardActivation class below
    NONE = auto() # apply no activation
    OUTPUT = auto() # apply an activation suitable for the output
                    # this will apply STANDARD activations for all
                    # except the very last layer

def crop(x, num=1) :
    """
    removes num rows on each side in each dimension of x
    x is assumed to have shape [batch, channel, x1, x2, x3]
    """
#{{{
    return x.narrow(2, num//2, x.shape[2]-num) \
            .narrow(3, num//2, x.shape[3]-num) \
            .narrow(4, num//2, x.shape[4]-num) \
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
        g = min(in_layout.channels, out_layout.channels)
        if in_layout.channels % g == 0 and out_layout.channels % g == 0 :
            return g
        else :
            return get_groups(GroupModes.ALL, None, None)
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

    def __repr__(self) :
        return '{ channels = %d, resolution = %d, density_channels = %d }'%(self.channels, self.resolution, self.density_channels)
#}}}

class Activation_BROKEN(nn.Module) :
    """
    our custom activation function that works on the channels we associate with density
    and on those we associate with displacement separately

    UPDATE : we don't seem to be able to work on channels separately...
             the backward function fails to compute gradients then.
             
             That's why I "commented out" this module, use the one below for now
             until we figure out how to do this
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

class StandardActivation(nn.LeakyReLU) :
    """
    The usual activation function.
    Initially we had Hardshrink here, since it was one of the only non-linearities
    that respected the sign symmetry, but Paco and Yin suggested that this is not actually
    that important and LeakyReLU may yield better training performance.
    """
#{{{
    def __init__(self, *args, **kwargs) :
        super().__init__(inplace=True, negative_slope=0.1)
#}}}

class OutputActivation(nn.Module) :
    """
    suitable for the output of the network, which should fall in the range [-1/2, 1/2]

    We want to use a periodic function so that the network has an easier time respecting
    the simulation periodicity.

    Note that we choose a relatively expensive function here,
    but since it is applied only once in the entire network it is completely fine.
    """
#{{{
    def __init__(self, *args, **kwargs) :
        super().__init__()

    def forward(self, x) :
        return 0.5 * torch.sin(x)
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

        # now that we have switched to size-2 kernels when resampling happens,
        # we don't need padding anymore in those cases
#        self.padding = 0 if self.resample is Resample.UP else 1
        self.padding = 1 if self.resample is Resample.EQUAL else 0

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
                          else nn.Conv3d)(in_layout.channels, out_layout.channels,
                                          # when up/downsampling, we use kernel size 2 to avoid checkerboard problems
                                          3 if resample is Resample.EQUAL else 2,
                                          stride=stride, groups=get_groups(group_mode, in_layout, out_layout),
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
        # -- only true for the size-3 kernels
#        if self.resample is Resample.UP :
#            x = crop(x)

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

        # TODO kaiming_uniform is only recommended for relu/leaky_relu activation
        #      functions
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
        # TODO kaiming_uniform is only recommended for relu/leaky_relu activation
        #      functions
        nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='leaky_relu')
        self.style_bias = nn.Parameter(torch.ones(in_chan))  # NOTE: init to 1

    def forward(self, x, s, eps=1e-8) :
        # TODO some of the batch-size dependent stuff could happen in the contructor

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

        # NOTE : for some weird reason torch complains about an in-place operation if I do the
        #        (in my understanding) completely equivalent w *= ... (as was there in the original code)
        #        Introducing the temporary w1 solves this problem.
        w1 = w * torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        # LFT : this piece of code is a bit tricky -- what it does is to have a different weight
        #       for each batch, still performing this operation in a single call
        w1 = w1.reshape(N * C0, C1, *K3)

        # LFT : added this, otherwise it doesn't work with non-unity batch sizes
        #       [self.bias] = [out_chan, ]
        if self.bias is not None :
            b1 = self.bias.expand(N, -1).flatten()
        else :
            b1 = None
        
        x = x.reshape(1, N * Cin, *DHWin)
        x = self.conv(x, w1, b1, groups=N)
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
                       activation=Activations.STANDARD, batch_norm=True, dropout=False, styles=False, bias=False,
                       group_mode=GroupModes.ALL, # do not change if styles=True
                       batch_norm_kw={}) :
        super().__init__()

        assert isinstance(activation, Activations)

        self.activation = (StandardActivation if activation is Activations.STANDARD
                           else OutputActivation if activation is Activations.OUTPUT
                           else nn.Identity)()
        self.batch_norm = (nn.Identity if not batch_norm else nn.BatchNorm3d) \
                                (out_layout.channels, **batch_norm_kw)
        self.dropout    = (nn.Identity if not dropout else nn.Dropout3d) \
                                (0.5 if isinstance(dropout, bool)
                                 else dropout if isinstance(dropout, float)
                                 else 0.5)

        resample=Resample.EQUAL if in_layout.resolution == out_layout.resolution \
                 else Resample.UP if 2*in_layout.resolution == out_layout.resolution \
                 else Resample.DOWN if in_layout.resolution == 2 * out_layout.resolution \
                 else None
        assert resample is not None, "Incompatible resolutions in=%d out=%d"%(in_layout.resolution,
                                                                              out_layout.resolution)
        common_conv_kwargs = dict(in_layout=in_layout, out_layout=out_layout,
                                  resample=resample, bias=bias)

        if styles :
            assert isinstance(styles, int)
            assert group_mode is GroupModes.ALL
            self.do_styles = True
            self.conv = StyleConv3d(N_styles=styles, **common_conv_kwargs)
        else :
            assert isinstance(styles, bool)
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
                   resampling     |                        |[ combined using +,
                     SINGLE       |           ALL          |  divide by sqrt(2), 
                    no style      |          styles        v  or concat (then extra collapse layer)]
    input[in_layout] ------> [out_layout] --> ... --> output[out_layout]
    """
#{{{
    def __init__(self, in_layout, out_layout,
                       N_layers=6, residual=True,
                       activation=Activations.STANDARD, batch_norm=False, dropout=False, bias=False,
                       batch_norm_kw={}) :
        super().__init__()

        self.residual = residual

        assert(N_layers > 1)
        common_layer_kwargs = dict(batch_norm=batch_norm, dropout=dropout,
                                   batch_norm_kw=batch_norm_kw)

        if out_layout.density_channels < 0 :
            out_layout.density_channels = in_layout.density_channels * out_layout.channels // in_layout.channels
        
        layers = []
        for ii in range(N_layers) :
            layers.append(Layer(in_layout=in_layout if ii==0 else out_layout,
                                out_layout=out_layout,
                                styles=False if ii==0 else settings.NSTYLES,
                                group_mode=GroupModes.SINGLE if ii==0 else GroupModes.ALL,
                                activation=activation \
                                           if activation is not Activations.OUTPUT or ii != N_layers-1 \
                                           else \
                                           Activations.OUTPUT,
                                bias=True if ii==0 else bias,
                                **common_layer_kwargs))

        if self.residual and not settings.RESIDUAL_ADD :
            concat_layout = Layout(channels=2*out_layout.channels,
                                   resolution=out_layout.resolution,
                                   density_channels=2*out_layout.density_channels)
            self.collapse = Layer(in_layout=concat_layout, out_layout=out_layout,
                                  group_mode=GroupModes.ALL,
                                  activation=activation,
                                  bias=True,
                                  **common_layer_kwargs)

        self.layers = nn.ModuleList(layers)

    def forward(self, x, s) :

        x = self.layers[0](x, s)
        
        if self.residual :
            xres = torch.clone(x)

        for _, l in enumerate(self.layers[1:]) :
            x = l(x, s)

        if self.residual :
            if settings.RESULTS_PATH :
                x += xres
                x /= math.sqrt(2) # against internal variance shift
            else :
                x = torch.stack((x, xres), dim=2).view(x.shape[0], 2*x.shape[1], *[x.shape[-1],]*3)
                x = self.collapse(x, s)

        return x
#}}}

class Level(nn.Module) :
    """
    Implements one level of a UNet architecture, looking something like


                             skip connection [optional]                 collapse [optional]
    input[in_layout]  ---------------------------------------------> x -------------------> output[out_layout]
        |                                                            ^
        |                                                            |
        | .contract                                                  | .expand
        v                                                            |
    output1[out_layout1]                                          input1[in_layout1]
             ... do something and pass the data to the right ...


    In the current implementation, the action is very rigid :
        layout       := in_layout = out_layout
        lower_layout := in_layout1 = out_layout1
    and out_layout1 will have the resolution halfed.

    Note that lower_layout will be available as members
    after construction, which can be used to construct lower levels.

    The only properties that can be controlled are by which factor
    the number of feature channels will be increased
    and whether there is a skip connection.
    The number of density channels will be changed proportionally

    Note also that the block properties at this level can be controlled using block_kw
    """
#{{{
    def __init__(self, layout, skip=True, channel_factor=2, **block_kw) :
        super().__init__()

        assert layout.resolution % 2 == 0
        self.lower_layout = Layout(channels=layout.channels*channel_factor, 
                                   resolution=layout.resolution // 2,
                                   density_channels=layout.density_channels*channel_factor)

        self.contract_block = Block(layout, self.lower_layout, **block_kw)

        self.expand_block = Block(self.lower_layout, layout, **block_kw)

        self.skip = skip
        # will hold the tensor for the skip connection in storage
        self.xskip = None

        # if we concatenate, we need to decrease the number of channels in the end
        # (note that in this case the level has slightly more parameters)
        if self.skip and not settings.SKIP_ADD :
            concat_layout = Layout(channels=2*layout.channels,
                                   resolution=layout.resolution,
                                   density_channels=2*layout.density_channels)
            self.collapse_block = Block(concat_layout, layout, N_layers=2, residual=False)

    def contract(self, x, s) :
        if self.skip :
            self.xskip = torch.clone(x)
        return self.contract_block(x, s)

    def expand(self, x, s) :
        x = self.expand_block(x, s)
        if self.skip :
            if settings.SKIP_ADD :
                x += self.xskip
                x /= math.sqrt(2) # against internal variance shift
            else :
                # we do the concatenation in an interleaving fashion so as to preserve
                # the order of density, x, y, z -- this is only important for operations
                # with non-trivial `groups' argument at the moment, but in general much
                # less confusing
                x = torch.stack((x, self.xskip), dim=2).view(x.shape[0], 2*x.shape[1], *[x.shape[-1],]*3)
                x = self.collapse_block(x, s)
            self.xskip = None

        return x
#}}}
