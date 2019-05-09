# coding:utf-8
from torch import nn
from collections import OrderedDict
from tensorboardX import SummaryWriter

net = {'21': [1, 1, 2, 2, 1],
       '53': [1, 2, 8, 8, 4]}


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channels[0], kernel_size=1,
                               stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):

    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.out_channel = 32
        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_channels = [64, 128, 256, 512, 1024]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out3, out4, out5

    def _make_layer(self, outputs, blocks):
        layers = list()
        # down sample
        layers.append(("conv", nn.Conv2d(self.out_channel, outputs[1], kernel_size=3,
                                         stride=2, padding=1)))
        layers.append(("bn", nn.BatchNorm2d(outputs[1])))
        layers.append(("relu", nn.LeakyReLU(0.1)))
        # blocks
        self.out_channel = outputs[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), ResidualBlock(self.out_channel, outputs)))
        return nn.Sequential(OrderedDict(layers))


def main():
    import numpy as np
    import torch
    input = torch.randn((32, 3, 256, 256))
    model = DarkNet(net['53'])
    print(model)

    with SummaryWriter() as w:
        w.add_graph(model, input)


if __name__ == '__main__':
    main()
