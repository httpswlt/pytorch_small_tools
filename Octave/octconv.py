# coding:utf-8
"""
    inference: https://export.arxiv.org/pdf/1904.05049
"""
from torch import nn


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        """

        :rtype: object
        """
        super(OctaveConv, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2"
        self.stride = stride
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1"
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else nn.Conv2d(
            int(alpha_in * in_channels), int(out_channels * out_channels), kernel_size=kernel_size, stride=1,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.conv_l2h = None if alpha_in == 0 and alpha_out == 1 else nn.Conv2d(
            int(alpha_in * in_channels), out_channels - int(out_channels * out_channels), kernel_size=kernel_size,
            stride=1, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 else nn.Conv2d(
            in_channels - int(alpha_in * in_channels), int(out_channels * out_channels), kernel_size=kernel_size,
            stride=1, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else nn.Conv2d(
            in_channels - int(alpha_in * in_channels), out_channels - int(out_channels * out_channels),
            kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 else None

        if x_l is not None:
            x_l2h = self.conv_l2h(x_l)
            x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None

            x_h = x_l2h + x_h2h
            x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
            return x_h, x_l
        else:
            return x_h2h, x_h2l


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(ConvBN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in=alpha_in, alpha_out=alpha_out,
                               stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_in == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()

        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in=alpha_in, alpha_out=alpha_out,
                               stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_in == 0 else norm_layer(int(out_channels * alpha_out))

        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l


class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, activation_layer=nn.ReLU):
        super(ConvAct, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in=alpha_in, alpha_out=alpha_out,
                               stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(x_h)
        x_l = self.act(x_l) if x_l is not None else None
        return x_h, x_l
