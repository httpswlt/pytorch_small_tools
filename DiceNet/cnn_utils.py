import torch
from torch import nn


def activation_fn(in_channel, name='prelu', inplace=True):
    """

    :param in_channel:
    :param name:
    :param inplace:
    :return:
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=inplace)
    elif name == 'prelu':
        return nn.PReLU(in_channel)
    else:
        NotImplementedError('Not implemented yet')
        exit()


class CBR(nn.Module):
    """
        This class defines the convolution layer with batch normalization and activation function
    """
    def __init__(self, in_channel, out_channel, ksize, stride=1, dilation=1, groups=1, act_name='prelu'):
        """

        :param in_channel: number of input channels
        :param out_channel: number of output channels
        :param ksize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        :param dilation:
        :param groups:  groups for group-wise convolution
        :param act_name: Name of the activation function
        """
        super(CBR, self).__init__()
        padding = int((ksize - 1) / 2) * dilation
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=ksize, stride=stride, padding=padding,
                      bias=False, groups=groups, dilation=dilation),
            nn.BatchNorm2d(out_channel),
            activation_fn(out_channel, name=act_name)
        )

    def forward(self, x):
        return self.cbr(x)


class CB(nn.Module):
    """
        This class implements convolution layer followed by batch normalization
    """

    def __init__(self, in_channel, out_channel, ksize, stride=1, dilation=1, groups=1):
        """
        :param in_channel: number of input channels
        :param out_channel: number of output channels
        :param ksize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        :param groups: # of groups for group-wise convolution
        """
        super(CB, self).__init__()
        padding = int((ksize - 1) / 2) * dilation
        self.cb = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, ksize, stride=stride, padding=padding, bias=False,
                      groups=groups, dilation=1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        return self.cb(x)


class BR(nn.Module):
    """
        This class implements batch normalization and  activation function
    """

    def __init__(self, in_channel, act_name='prelu'):
        """
        :param in_channel: number of input channels
        :param act_name: Name of the activation function
        """
        super(BR, self).__init__()
        self.br = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            activation_fn(in_channel, name=act_name)
        )

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        return self.br(x)


class Shuffle(nn.Module):
    """
    This class implements Channel Shuffling
    """
    def __init__(self, groups):
        """
        :param groups: groups for shuffling
        """
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        bs, channel, h, w = x.data.size()
        channel_per_group = channel // self.groups
        x = x.view((bs, self.groups, channel_per_group, h, w))
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view((bs, -1, h, w))
        return x


class DWConv(nn.Module):
    def __init__(self, in_channel):
        super(DWConv, self).__init__()
        self.dw_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.PReLU(in_channel)
        )

    def forward(self, x):
        return self.dw_layer(x)
