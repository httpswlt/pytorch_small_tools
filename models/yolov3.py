# coding: utf-8
import torch
from torch import nn
from collections import OrderedDict
from backbone.darknet import DarkNet, net

parameter = {"yolo": {
                        "anchors": [[[116, 90], [156, 198], [373, 326]],
                                    [[30, 61], [62, 45], [59, 119]],
                                    [[10, 13], [16, 30], [33, 23]]],
                        "classes": 21,
                    },
            }


class YOLOV3(nn.Module):
    def __init__(self, config):
        super(YOLOV3, self).__init__()
        self.backbone = DarkNet(net['53'])
        backbone_output = self.backbone.layers_out_channels

        # embedding 0
        final_out_channel0 = len(config["yolo"]["anchors"][0] * (5 + config["yolo"]["classes"]))
        self.embedding0 = self._make_embedding([512, 1024], backbone_output[-1], final_out_channel0)

        # embedding 1
        final_out_channel1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.embedding1_conv_block = self.conv2d_block(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2)
        self.embedding1 = self._make_embedding([256, 512], backbone_output[-2] + 256, final_out_channel1)

        # embedding 2
        final_out_channel2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.embedding2_conv_block = self.conv2d_block(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2)
        self.embedding2 = self._make_embedding([128, 256], backbone_output[-3] + 128, final_out_channel2)

    def forward(self, x):
        def _branch(input, embedding):
            branch_out = None
            for i, conv in enumerate(embedding):
                input = conv(input)
                if i == 4:
                    branch_out = input
            return branch_out, input

        # backbone
        x2, x1, x0 = self.backbone(x)

        # yolo branch 0 output
        embed_out_0, pred_out_0 = _branch(x0, self.embedding0)

        # yolo branch 1 output
        x1_in = self.embedding1_conv_block(embed_out_0)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat((x1_in, x1), 1)
        embed_out_1, pred_out_1 = _branch(x1_in, self.embedding1)

        # yolov1 branch 2 output
        x2_in = self.embedding2_conv_block(embed_out_1)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat((x2_in, x2), 1)
        embed_out_2, pred_out_2 = _branch(x2_in, self.embedding2)

        return pred_out_0, pred_out_1, pred_out_2

    def _make_embedding(self, out_channels, in_channel, final_out_channel):
        layer = nn.ModuleList([
            self.conv2d_block(in_channel, out_channels[0], 1),
            self.conv2d_block(out_channels[0], out_channels[1], 3),
            self.conv2d_block(out_channels[1], out_channels[0], 1),
            self.conv2d_block(out_channels[0], out_channels[1], 3),
            self.conv2d_block(out_channels[1], out_channels[0], 1),
            self.conv2d_block(out_channels[0], out_channels[1], 3),
        ])
        layer.add_module("conv_out", nn.Conv2d(out_channels[1], final_out_channel, kernel_size=1))
        return layer

    @staticmethod
    def conv2d_block(in_channel, out_channel, kernel_size, stride=1, padding=None):
        if not padding:
            padding = (kernel_size - 1) // 2 if kernel_size else 0
        return nn.Sequential(OrderedDict([
                    ("conv", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                                       stride=stride, padding=padding)),
                    ("bn", nn.BatchNorm2d(out_channel)),
                    ("relu", nn.LeakyReLU(0.1)),
               ]))


def main():
    model = YOLOV3(parameter)
    x = torch.randn((1, 3, 416, 416))
    y0, y1, y2 = model(x)
    print(y0.size())
    print(y1.size())
    print(y2.size())


if __name__ == '__main__':
    main()
