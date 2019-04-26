# coding:utf-8
from torch import nn


class VGG(nn.Module):
    cfg = {
        11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, depth=16, bn=False, num_classes=1000):
        super(VGG, self).__init__()
        self.structs = self.cfg[depth]
        self.bn = bn
        self.feature_extract = self.make_layers()
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
        x = self.feature_extract(x)
        if self.is_classify:
            x = x.view(x.size(0), -1)
            self.classifier(x)
        return x

    def make_layers(self):
        layers = []
        in_channel = 3
        for v in self.structs:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, v, kernel_size=3, padding=1)
                if self.bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = v
        return nn.Sequential(*layers)


if __name__ == '__main__':
    vgg16 = VGG(16, num_classes=0).cuda()

