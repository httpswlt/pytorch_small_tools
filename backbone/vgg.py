# coding:utf-8
from torch import nn


class VGG(nn.Module):

    def __init__(self, structs, bn=False, num_classes=1000):
        """

        :param depth: network depth.
        :param bn: whether enable batch normal.
        :param num_classes: when value is 0, the feature extractor of detection network.
                            ,otherwise, it's a classify network.
        """
        super(VGG, self).__init__()
        self.structs = structs
        self.bn = bn
        self.feature_extract = nn.ModuleList(self.make_layers(self.structs, self.bn))
        self.is_classify = num_classes
        if self.is_classify:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        for i in range(len(self.feature_extract)):
            x = self.feature_extract[i](x)
        if self.is_classify:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

    @staticmethod
    def make_layers(structs, bn=False):
        layers = []
        in_channel = 3
        for v in structs:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channel, v, kernel_size=3, padding=1)
                if bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = v
        return layers


def vgg16(num_classes=0):
    cfg = {
        11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    return VGG(cfg[16], num_classes=num_classes).cuda()


if __name__ == '__main__':
    vgg16()
