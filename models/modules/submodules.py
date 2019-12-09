import torch
import torch.nn as nn

def act(act_type, **kwargs):
    """
    Helper to select activation layer with string
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace=False, **kwargs)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(0.2, inplace=False, **kwargs)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=64, init=0.2, **kwargs)
    else:
        raise NotImplementedError(f'Activation layer [{act_type}] is not found')
    return layer

def norm(norm_type, nc):
    """
    Helper to select normalization layer with string
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return layer

def pad(pad_type, kernel_size=None, exact_pad_size=None):
    """
    Helper to select padding layer with string
    It also infers suitable pad size with kernel_size if exact_pad is not given.
    exact_pad overrides kernel_size.
    """
    pad_type = pad_type.lower()
    if kernel_size:
        pad_size = (kernel_size - 1) // 2
    if exact_pad_size:
        pad_size = exact_pad_size
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(pad_size)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(pad_size)
    elif pad_type == 'zero':
        layer = nn.ZeroPad2d(pad_size)
    else:
        raise NotImplementedError(f'padding layer [{pad_type}] is not implemented')
    return layer

def identity(inputs):
    """
    Dummy function for simplifying syntax
    """
    return inputs

class ShortcutBlock(nn.Module):
    """
    Elementwise sum the input of a submodule to its output
    """
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.submodule = submodule

    def forward(self, inputs):
        output = inputs + self.submodule(inputs)
        return output

    def __repr__(self):
        tmpstr =  'Identity + \n|'
        modstr = self.submodule.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

class SumBlock(nn.Module):
    """
    Elementwise sum the 2 inputs
    """
    def __init__(self):
        super(SumBlock, self).__init__()

    def forward(self, input1, input2):
        output = input1 + input2
        return output

    def __repr__(self):
        tmpstr =  '+ \n|'
        return tmpstr

class Conv2dBlock(nn.Module):
    """
    Conv2d Layer with padding, normalization, activation, dropout
    """
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 pad_type='reflect', norm_type=None, act_type='relu', use_dropout=False):
        super(Conv2dBlock, self).__init__()
        self.P = pad(pad_type, kernel_size) if pad_type else identity
        self.C = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.N = norm(norm_type, output_nc) if norm_type else identity
        self.A = act(act_type) if act_type else identity
        self._initialize_weights()

    def forward(self, x):
        x = self.P(x)
        x = self.C(x)
        x = self.N(x)
        x = self.A(x)
        return x

    def _initialize_weights(self):
        # if isinstance(self.A, nn.LeakyReLU):
        #     a = 0.2 # LeakyReLU default negative slope
        # elif isinstance(self.A, nn.PReLU):
        #     a = 0.2 # PReLU default initial negative slope
        # elif isinstance(self.A, nn.ReLU):
        #     a = 0.0 # ReLU has zero negative slope
        # else:
        #     a = 1.0
        
        nn.init.normal_(self.C.weight, mean=0.0, std=0.02)
        if self.C.bias is not None:
            nn.init.constant_(self.C.bias, val=1.0)

        if isinstance(self.N, nn.BatchNorm2d):
            nn.init.constant_(self.N.weight, val=1.0)
            if self.N.bias is not None:
                nn.init.constant_(self.N.bias, val=0.0)

def ResNetBlock(
    input_nc, 
    mid_nc, 
    output_nc, 
    kernel_size=3, 
    stride=1, 
    bias=True, 
    pad_type='reflect', 
    norm_type='batch', 
    act_type='prelu', 
    use_dropout=False
):
    conv1 = Conv2dBlock(
        input_nc, 
        mid_nc,
        kernel_size, 
        stride, 
        bias=bias, 
        pad_type=pad_type, 
        norm_type=norm_type, 
        act_type=act_type, 
        use_dropout=use_dropout
    )
    conv2 = Conv2dBlock(
        mid_nc, 
        output_nc, 
        kernel_size, 
        stride, 
        bias=bias, 
        pad_type=pad_type, 
        norm_type=norm_type, 
        act_type=None, 
        use_dropout=False
    )
    residual_features = nn.Sequential(conv1, conv2)
    return ShortcutBlock(residual_features)

class SubPixelConvBlock(nn.Module):
    def __init__(
        self, 
        input_nc, 
        output_nc, 
        feature_factor=2, 
        kernel_size=3, 
        stride=1, 
        bias=True, 
        pad_type='reflect', 
        norm_type=None, 
        act_type='prelu', 
        use_dropout=False
    ):
        super(SubPixelConvBlock, self).__init__()
        self.conv_block = Conv2dBlock(
            input_nc, 
            output_nc * (feature_factor ** 2), 
            kernel_size, 
            stride, 
            bias=bias, 
            pad_type=pad_type, 
            norm_type=norm_type, 
            act_type=None, 
            use_dropout=use_dropout
        )
        self.PS = nn.PixelShuffle(feature_factor)
        self.A = act(act_type) if act_type else identity

    def forward(self, inputs):
        output = self.conv_block(inputs)
        output = self.PS(output)
        output = self.A(output)
        return output