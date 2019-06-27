import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gc
import encodings


__all__ = ['Darknet53', 'darknet_53']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def darknet_53(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # if pretrained:
    #     if 'transform_input' not in kwargs:
    #         kwargs['transform_input'] = True
    #     model = Darknet53(**kwargs)
    #     model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
    #     return model
    model = Darknet53([1, 2, 8, 8, 4], **kwargs)

    return model


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']
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


class EmptyLayer(nn.Module):
    __slots__ = []

    def __init__(self):
        super(EmptyLayer, self).__init__()


class MaxPoolStride1(nn.Module):
    __slots__ = ['kernel_size','pad']

    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class DetectionLayer(nn.Module):
    __slots__ = ['anchors']

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks, input_size=256):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()   # create a model container contain all layer.
    in_channel = 3
    output_channels = []
    downsize_stage = 0
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()  # create a sequential container by order.
        if x["type"] == "convolutional":    # If it's a convolutional layer
            activation = x["activation"]    # Get the info about the layer# Get the info about the layer
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            out_channel = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            if stride == 2:
                downsize_stage += 1
            pad = (kernel_size - 1) // 2 if padding else 0
            conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad, bias=bias)    # Add the convolutional layer
            module.add_module("conv_{0}".format(index), conv)
            if batch_normalize:    # Add the Batch Norm Layer
                bn = nn.BatchNorm2d(out_channel)
                module.add_module("batch_norm_{0}".format(index), bn)
            if activation == "leaky":   # Add the activation function
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
        elif x["type"] == "upsample":    # If it's an upsampling layer
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")    # We can also use the bilinear
            module.add_module("upsample_{}".format(index), upsample)
        elif x["type"] == "route":     # statement route,then we will implement.
            layers = x["layers"].split(',')
            layers = [int(l) if int(l) > 0 else int(l) + index for l in layers]
            nums = len(layers)
            if nums == 1:
                out_channel = output_channels[layers[0]]
            else:
                out_channel = output_channels[layers[0]] + output_channels[layers[1]]
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

        elif x["type"] == "shortcut":   # statement shorcut
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            module.add_module("maxpool_{}".format(index), maxpool)

        elif x["type"] == "yolo":   # Yolo is the detection layer
            mask = x["mask"].split(",")
            mask = [int(i) for i in mask]
            anchors = x["anchors"].split(",")
            anchors = [float(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        elif x["type"] == "avgpool":   # Yolo is the detection layer
            avgpool = nn.AvgPool2d(kernel_size=int(input_size/pow(2, downsize_stage)), stride=1)
            module.add_module("avgpool_{}".format(index), avgpool)
        elif x["type"] == "softmax":   # statement shorcut
            softmax = EmptyLayer()
            module.add_module("shortcut_{}".format(index), softmax)
        else:
            print("Something I dunno")
            assert False
        module_list.append(module)
        in_channel = out_channel
        output_channels.append(out_channel)
    return net_info, module_list


class Darknet_cfg_53(nn.Module):

    def __init__(self, cfgfile,  num_classes):
        super(Darknet_cfg_53, self).__init__()
        self.blocks = parse_cfg(cfgfile)    # all blocks from cfg file
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        self.num_classes = num_classes
        self.classification = []
        self.downsize_stage = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def model_name(self):
        return self.__class__.__name__

    def forward(self, x):
        block_list = self.blocks[1:]    # remove net info layer.
        outputs = {}  # We cache the outputs for the route layer
        # is_first_yolo = 0
        for i, block in enumerate(block_list):
            module_type = (block["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
                outputs[i] = x
            elif module_type == "avgpool":
                x = self.module_list[i](x)
                outputs[i] = x
            elif module_type == "route":
                layers = block["layers"].split(',')
                layers = [int(l) if int(l) > 0 else int(l) + i for l in layers]
                nums = len(layers)
                if nums == 1:
                    x = outputs[layers[0]]
                elif nums == 2:
                    map1 = outputs[layers[0]]
                    map2 = outputs[layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x
            elif module_type == "shortcut":
                from_ = int(block["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x
            elif module_type == 'softmax':
                self.classification = x.view(x.size(0), -1)
                outputs[i] = outputs[i - 1]
        return self.classification

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)    # The rest of the values are the weights
        ptr = 0
        for i, model in enumerate(self.module_list):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    # Number of biases
                    num_biases = conv.bias.numel()
                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
        fp.close()

    def __del__(self):
        del self.blocks
        del self.net_info
        del self.module_list
        del self.header
        del self.seen
        del self.num_classes
        del self.classification
        gc.collect()
        torch.cuda.empty_cache()


class Darknet53(nn.Module):

    transforms_para = transforms.Compose([
            transforms.RandomResizedCrop(256),
            # transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
        ])

    burn_in = 2000
    power = 4

    def model_name(self):
        return self.__class__.__name__

    def __init__(self, layers, input_size=256, num_classes=1000):
        super(Darknet53, self).__init__()
        self.conv1 = BasicConv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.layer5 = self._make_layer(1024, layers[4], stride=2)
        last_channels = int(input_size/pow(2, len(layers)))
        self.avgpool = nn.AvgPool2d(kernel_size=last_channels, stride=1)
        self.classifier = nn.Conv2d(1024, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=2):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                BasicConv2d(in_channels=planes//2, out_channels=planes, kernel_size=3, stride=2)
            )

        layers = []
        layers.append(Bottleneck(in_channels=planes, downsample=downsample))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(Bottleneck(planes))

        return nn.Sequential(*layers)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = x.view(x.size(0), -1)
        return x


class Bottleneck(nn.Module):

    expansion = 2

    def __init__(self, in_channels, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1x1 = BasicConv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv3x3 = BasicConv2d(in_channels // 2, in_channels, kernel_size=3)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        residual = x
        output = self.conv1x1(x)
        output = self.conv3x3(output)

        output = output + residual
        return output


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, 0.1, inplace=True)
