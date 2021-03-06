# coding:utf-8
import sys
import torch
from torch import nn
import torch.nn.functional as F
sys.path.append("../")
from backbone.vgg import VGG
from module.l2norm import L2Norm


class SSD(nn.Module):
    def __init__(self, image_size=300, num_classes=10, phase="train"):
        super(SSD, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.phase = phase
        vgg_struct, extra_struct, mboxs = self.configure(str(self.image_size))

        self.vgg = VGG.make_layers(vgg_struct)
        # remove the last three full layers of vgg, then add two convolution layers
        self.vgg += [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)]
        self.vgg = nn.ModuleList(self.vgg)

        self.extras = nn.ModuleList(self.add_extra_layer(extra_struct))

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

        # predict location and classification layers
        self.loc, self.conf = (nn.ModuleList(temp) for temp in self.detector_layer(mboxs))

    def forward(self, x):
        sources = list()    # it contain some layers what will predict some location and classification
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([l.view(l.size(0), -1) for l in loc], 1)
        conf = torch.cat([c.view(c.size(0), -1) for c in conf], 1)

        if self.phase == "test":
            # pass
            output = None
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes)
            )
        return output

    def detector_layer(self, mboxs):
        loc_layers = []
        conf_layers = []
        vgg_source = [21, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(self.vgg[v].out_channels,
                                     mboxs[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.vgg[v].out_channels,
                                      mboxs[k] * self.num_classes, kernel_size=3, padding=1)]

        for k, v in enumerate(self.extras[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, mboxs[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mboxs[k]
                                      * self.num_classes, kernel_size=3, padding=1)]
        return loc_layers, conf_layers

    @staticmethod
    def configure(image_size):
        mbox = {
            '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
        }
        extras = {
            '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        }
        vgg_cfg = {
            '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                    512, 512, 512],
        }
        return vgg_cfg[image_size], extras[image_size], mbox[image_size]

    @staticmethod
    def add_extra_layer(structs):
        layers = []
        in_channels = 1024
        # add extra layers.
        flag = True
        for k, v in enumerate(structs):
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, structs[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        return layers


