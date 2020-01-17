import torch
import torch.nn as nn
import torch.nn.functional as F
## We use the batchnorm in pytorch 0.3.x to avoid unexpected errors
## Otherwise, some in-place modification will bring some errors
from models.utils.batchnorm import BatchNorm1d, BatchNorm2d



def get_normalization_2d(channels, normalization):
    if normalization == 'instance':
        return nn.InstanceNorm2d(channels)
    elif normalization == 'batch':
        return BatchNorm2d(channels)
    elif normalization == 'none':
        return None
    else:
        raise ValueError('Unrecognized normalization type "%s"' %
                         normalization)


def get_activation(name):
    kwargs = {}
    if name.lower().startswith('leakyrelu'):
        if '-' in name:
            slope = float(name.split('-')[1])
            kwargs = {'negative_slope': slope}
            name = 'leakyrelu'
    activations = {
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
    }
    if name.lower() not in activations:
        raise ValueError('Invalid activation "%s"' % name)
    return activations[name.lower()](**kwargs)


def weights_init(m):
    raise NotImplementedError
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm1d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return 'Flatten()'


class Unflatten(nn.Module):
    def __init__(self, size):
        super(Unflatten, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(*self.size)

    def __repr__(self):
        size_str = ', '.join('%d' % d for d in self.size)
        return 'Unflatten(%s)' % size_str


class GlobalAvgPool(nn.Module):
    def __init__(self, pooling="avg"):
        super(GlobalAvgPool, self).__init__()
        self.pooling = pooling
    def forward(self, x):
        N, C = x.size(0), x.size(1)
        if self.pooling == "avg":
            return x.view(N, C, -1).mean(dim=2)
        elif self.pooling == "max":
            return x.view(N, C, -1).max(dim=2)[0]
        else:
            assert False, "Unrecognized pooling: {}".format(pooling)


class ResidualBlock(nn.Module):
    def __init__(self, channels, normalization='batch', activation='relu',
                 padding='same', kernel_size=3, init='default'):
        super(ResidualBlock, self).__init__()
        K = kernel_size
        P = _get_padding(K, "same")
        C = channels
        self.padding = P
        layers = [
            get_normalization_2d(C, normalization),
            get_activation(activation),
            nn.Conv2d(C, C, kernel_size=K, padding=P),
            get_normalization_2d(C, normalization),
            get_activation(activation),
            nn.Conv2d(C, C, kernel_size=K, padding=P),
        ]
        layers = [layer for layer in layers if layer is not None]
        # for layer in layers:
        #     _init_conv(layer, method=init)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        P = self.padding
        shortcut = x
        y = self.net(x)
        return shortcut + self.net(x)


def _get_padding(K, mode):
    """ Helper method to compute padding size """
    if mode == 'valid':
        return 0
    elif mode == 'same':
        assert K % 2 == 1, 'Invalid kernel size %d for "same" padding' % K
        return (K - 1) // 2


def build_cnn(arch, normalization='batch', activation='relu', padding='same',
              pooling='max', init='default', non_linear_activated=False):
    """
    Build a CNN from an architecture string, which is a list of layer
    specification strings. The overall architecture can be given as a list or as
    a comma-separated string.

    All convolutions *except for the first* are preceeded by normalization and
    nonlinearity.

    non_linear_activated: whether to add [non-linear activation / normalization]
        Default is [False] (do not add these layers)

    All other layers support the following:
    - IX: Indicates that the number of input channels to the network is X.
          Can only be used at the first layer; if not present then we assume
          3 input channels.
    - CK-X: KxK convolution with X output channels
    - CK-X-S: KxK convolution with X output channels and stride S
    - R: Residual block keeping the same number of channels
    - UX: Nearest-neighbor upsampling with factor X
    - PX: Spatial pooling with factor X
    - FC-X-Y: Flatten followed by fully-connected layer

    Returns a tuple of:
    - cnn: An nn.Sequential
    - channels: Number of output channels
    """
    if isinstance(arch, str):
        arch = arch.split(',')
    cur_C = 3
    if len(arch) > 0 and arch[0][0] == 'I':
        cur_C = int(arch[0][1:])
        arch = arch[1:]

    first_conv = True
    flat = False
    layers = []
    for i, s in enumerate(arch):
        if s[0] == 'C':
            if not first_conv:
                layers.append(get_normalization_2d(cur_C, normalization))
                layers.append(get_activation(activation))
            first_conv = False
            vals = [int(i) for i in s[1:].split('-')]
            if len(vals) == 2:
                K, next_C = vals
                stride = 1
            elif len(vals) == 3:
                K, next_C, stride = vals
            # K, next_C = (int(i) for i in s[1:].split('-'))
            P = _get_padding(K, padding)
            conv = nn.Conv2d(cur_C, next_C, kernel_size=K,
                             padding=P, stride=stride)
            layers.append(conv)
            # _init_conv(layers[-1], init)
            cur_C = next_C
        elif s[0] == 'R':
            norm = 'none' if first_conv else normalization
            res = ResidualBlock(cur_C, normalization=norm, activation=activation,
                                padding=padding, init=init)
            layers.append(res)
            first_conv = False
        elif s[0] == 'U':
            factor = int(s[1:])
            layers.append(nn.Upsample(scale_factor=factor, mode='nearest'))
        elif s[0] == 'P':
            factor = int(s[1:])
            if pooling == 'max':
                pool = nn.MaxPool2d(kernel_size=factor, stride=factor)
            elif pooling == 'avg':
                pool = nn.AvgPool2d(kernel_size=factor, stride=factor)
            layers.append(pool)
        elif s[:2] == 'FC':
            _, Din, Dout = s.split('-')
            Din, Dout = int(Din), int(Dout)
            if not flat:
                layers.append(Flatten())
            flat = True
            layers.append(nn.Linear(Din, Dout))
            if i + 1 < len(arch):
                layers.append(get_activation(activation))
            cur_C = Dout
        else:
            raise ValueError('Invalid layer "%s"' % s)
    if non_linear_activated:
        layers.append(get_normalization_2d(cur_C, normalization))
        layers.append(get_activation(activation))
    layers = [layer for layer in layers if layer is not None]
    for layer in layers:
        print(layer)

    model = nn.Sequential(*layers)
    return model, cur_C


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True,
              start_nonlinearity=False,):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        if start_nonlinearity and i == 0:
            if batch_norm == 'batch':
                layers.append(BatchNorm1d(dim_in))
            # elif batch_norm == 'instance':
            #     layers.append(InstanceNorm1d(dim_in))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    model = nn.Sequential(*layers)
    return model
