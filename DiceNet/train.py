"""
    inference: https://arxiv.org/pdf/1906.03516.pdf
"""

import torch
from torch import nn
from cnn_utils import CBR

sc_ch_dict = {
            #0.1 : [8, 8, 16, 32, 64, 512],
            0.2: [16, 16, 32, 64, 128, 1024],
            0.5: [24, 24, 48, 96, 192, 1024],
            0.75: [24, 24, 86, 172, 344, 1024],
            1.0: [24, 24, 116, 232, 464, 1024],
            1.25: [24, 24, 144, 288, 576, 1024],
            1.5: [24, 24, 176, 352, 704, 1024],
            1.75: [24, 24, 210, 420, 840, 1024],
            2.0: [24, 24, 244, 488, 976, 1024],
            2.4: [24, 24, 278, 556, 1112, 1280],
            #3.0: [48, 48, 384, 768, 1536, 2048]
        }
rep_layers = [0, 3, 7, 3]


class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()

        # ====================
        # Network configuration
        # ====================
        num_classes = args.num_classes
        in_channel = args.channels
        width = args.model_width
        height = args.model_height
        s = args.s

        if s not in sc_ch_dict.keys():
            print('Model at scale s={} is not suppoerted yet'.format(s))
            exit(0)

        out_channel_map = sc_ch_dict[s]
        reps_at_each_level = rep_layers

        assert width % 32 == 0, 'Input image width should be divisible by 32'
        assert height % 32 == 0, 'Input image height should be divisible by 32'

        # ====================
        # Network architecture
        # ====================
        # output size will be 112 x 112
        width = int(width / 2)
        height = int(height / 2)
        self.level1 = CBR(in_channel, out_channel_map[0], 3, 2)
        width = int(width / 2)
        height = int(height / 2)
        self.level2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # output size will be 28 x 28
        width = int(width / 2)
        height = int(height / 2)
        level3 = nn.ModuleList()


    def forward(self, *input):
        pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()

    for scale in sc_ch_dict.keys():
        for size in [224]:
            # args.num_classes = 1000
            imSz = size
            args.s = scale
            args.channels = 3
            args.model_width = 224
            args.model_height = 224
            args.num_classes = 1000
            model = CNNModel(args)
            # input = torch.randn(1, 3, size, size)
            # print_info_message('Scale: {}, ImSize: {}'.format(scale, size))
            # print_info_message('Flops: {:.2f} million'.format(compute_flops(model, input)))
            # print_info_message('Params: {:.2f} million'.format(model_parameters(model)))
            # print('\n')
            # exit()