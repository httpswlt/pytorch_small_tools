# coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

"""
    For RetinaNet
"""


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks, bias=False):
        super(FPN, self).__init__()
        self.in_channels = 64
        self.bias = bias

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=bias)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayers1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayers2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayers3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Bottom-up
        # ResNet 50 layer construct
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # add extra double layers to predict
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        # Top-down, Pyramid Feature construct
        p5 = self.latlayers1(c5)
        p4 = self._upsample_add(p5, self.latlayers2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayers3(c3))
        p3 = self.toplayer2(p3)
        return p3, p4, p5, p6, p7

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.bias))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    @staticmethod
    def _upsample_add(x, y):
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear') + y


def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])


def FPN101():
    return FPN(Bottleneck, [2, 4, 23, 3])


def test():
    x = torch.zeros((2, 3, 600, 300))
    model = FPN50()
    print(model)
    a = model(x)


if __name__ == '__main__':
    test()
