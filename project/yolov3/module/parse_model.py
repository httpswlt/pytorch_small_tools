# coding:utf-8
from torch import nn
from modules import EmptyLayer, MaxPoolStride1, YOLOLayer, Upsample


def create_modules(blocks):
    """
        Constructs module list of layer blocks from module configuration in module_defs
    """
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()  # create a model container contain all layer.
    in_channel = 3
    output_channels = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x["type"] == "convolutional":

            activation = x["activation"]
            batch_normalize = int(x.get("batch_normalize", 0))
            out_channel = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            pad = (kernel_size - 1) // 2 if padding else 0
            module.add_module(
                "conv_{0}".format(index),
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    stride,
                    pad,
                    bias=bool(batch_normalize)
                )
            )

            if batch_normalize:
                module.add_module("batch_norm_{0}".format(index),
                                  nn.BatchNorm2d(out_channel))

            if activation == "leaky":
                module.add_module("leaky_{0}".format(index),
                                  nn.LeakyReLU(0.1, inplace=True))

        elif x["type"] == "upsample":
            module.add_module("upsample_{}".format(index),
                              Upsample(scale_factor=int(x["stride"]), mode="nearest"))

        elif x["type"] == "route":
            layers = x["layers"].split(',')
            layers = [int(l) if int(l) > 0 else int(l) + index for l in layers]
            nums = len(layers)
            if nums == 1:
                out_channel = output_channels[layers[0]]
            else:
                out_channel = output_channels[layers[0]] + output_channels[layers[1]]
            module.add_module("route_{0}".format(index), EmptyLayer())

        elif x["type"] == "shortcut":
            module.add_module("shortcut_{}".format(index), EmptyLayer())

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            module.add_module("maxpool_{}".format(index), maxpool)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(i) for i in mask]
            anchors = x["anchors"].split(",")
            anchors = [float(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = YOLOLayer(anchors, int(x['classes']),
                                  (net_info['width'], net_info['height']))
            module.add_module("Detection_{}".format(index), detection)
        else:
            print("Something I dunno")
            assert False
        module_list.append(module)
        in_channel = out_channel
        output_channels.append(out_channel)

    return net_info, module_list


def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']
    lines = [x.split('#')[0] for x in lines]
    lines = [x.strip() for x in lines]
    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks
