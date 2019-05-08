# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import parse_cfg,predict_transform,reorg,region_predict_transform
import gc

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

   
class ReorgLayer(nn.Module):
    __slots__ = ['stride', 'rearrange_index']
    
    def __init__(self, stride, in_channel, reorg_size):
        super(ReorgLayer, self).__init__()
        self.stride = stride
        C = in_channel
        H = reorg_size
        self.rearrange_index = np.arange(stride*C*H)
        for i in range(stride*C*H):
            if i % 2 == 0:
                if i < stride*C*H / 2:
                    self.rearrange_index[i+1] = self.rearrange_index[i] + stride*C*H / 2
                else:
                    self.rearrange_index[i] = i - stride*C*H / 2 + 1
                    self.rearrange_index[i+1] = self.rearrange_index[i] + stride*C*H / 2
    
    def forward(self, x):
        input = x.data
        B,C,H,W = input.size()
        tmp = input.view(B, C, H/self.stride, self.stride, W/self.stride, self.stride).transpose(3,4).contiguous()
        tmp = tmp.view(B, C, H/self.stride*W/self.stride, self.stride*self.stride).transpose(2,3).contiguous()
        tmp = tmp.view(B, C, self.stride*self.stride, H/self.stride, W/self.stride).transpose(1,2).contiguous()
        tmp = tmp.view(B, self.stride*C*H, W/self.stride)

        tmp[0] = tmp[0][self.rearrange_index]
        result = tmp.view(B, self.stride*self.stride*C, H/self.stride, W/self.stride)

        return result 


class YOLO_V3_Pytorch(nn.Module):
    def __init__(self, cfgfile, gpu_num, num_classes, fp16):
        super(YOLO_V3_Pytorch, self).__init__()
        self.blocks = parse_cfg(cfgfile)    # all blocks from cfg file
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.num_classes = num_classes
        self.gpu_num = gpu_num
        self.fp16 = fp16

    def forward(self, x):
        self.detections = []
        block_list = self.blocks[1:]    # remove net info layer.
        outputs = {}  # We cache the outputs for the route layer
        is_first_yolo = 0
        for i,block in enumerate(block_list):
            module_type = (block["type"])
            if module_type == "convolutional" or module_type == "deconv" \
               or module_type == "upsample" or module_type == "maxpool" \
               or module_type == 'reorg':
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
                    x = torch.cat((map1,map2),1)
                outputs[i] = x
            elif module_type == "shortcut":
                from_ = int(block["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                im_dim = (int(self.net_info["width"]), int(self.net_info["height"]))
                num_classes = int(block["classes"])    # Get the number of classes
                x = predict_transform(x.data, im_dim, anchors, num_classes, self.gpu_num)    # Output the result
                if not is_first_yolo:
                    self.detections = x
                    is_first_yolo = 1
                else:
                    self.detections = torch.cat((self.detections, x), 1)
                outputs[i] = outputs[i - 1]
            elif module_type == 'region':
                anchors = self.module_list[i][0].anchors
                im_dim = (int(self.net_info["width"]), int(self.net_info["height"]))
                num_classes = int(block["classes"])    # Get the number of classes
                self.detections = region_predict_transform(x.data, im_dim, anchors, num_classes, self.gpu_num)

        return self.detections

    def get_predict(self, conf):
        max_conf, _ = torch.max(self.detections[:, :, 5:5 + self.num_classes], 2)
        self.detections[:, :, 4] *= max_conf
        conf_mask = (self.detections[:, :, 4] > conf).float().unsqueeze(2)
        if self.fp16:
            conf_mask = conf_mask.half()
        prediction = self.detections * conf_mask
        box_a = prediction.new(prediction.shape)
        box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
        box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
        box_a[:, :, 2] = prediction[:, :, 2]
        box_a[:, :, 3] = prediction[:, :, 3]
        prediction[:, :, :4] = box_a[:, :, :4]
        image_pred = prediction[0]
        max_conf, max_conf_index = torch.max(image_pred[:, 5:5 + self.num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        if self.fp16:
            max_conf = max_conf.half()
        seq = (image_pred[:, :5], max_conf)
        image_pred = torch.cat(seq, 1)
        # Get rid of the zero entries
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 6)
        if self.fp16:
            image_pred_ = image_pred_.float()
        return image_pred_

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)    # The rest of the values are the weights
        ptr = 0
        for i,model in enumerate(self.module_list):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional" or module_type == "deconv":
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
        del self.gpu_num
        del self.detections
        gc.collect()
        torch.cuda.empty_cache()


def create_modules(blocks):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()   # create a model container contain all layer.
    in_channel = 3
    output_channels = []
    for index,x in enumerate(blocks[1:]):
        module = nn.Sequential()  # create a sequential container by order.
        if (x["type"] == "convolutional" or x["type"] == "deconv"):    # If it's a convolutional layer
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
            pad = (kernel_size - 1) // 2 if padding else 0

            if x["type"] == "convolutional":
                conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad, bias=bias)    # Add the convolutional layer
                module.add_module("conv_{0}".format(index), conv)
            else:
                deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, pad, bias=bias)
                module.add_module("deconv_{0}".format(index), deconv)

            if batch_normalize:    # Add the Batch Norm Layer
                bn = nn.BatchNorm2d(out_channel)
                module.add_module("batch_norm_{0}".format(index), bn)
            if activation == "leaky":   # Add the activation function
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
        elif (x["type"] == "upsample"):    # If it's an upsampling layer
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")    # We can also use the bilinear
            module.add_module("upsample_{}".format(index), upsample)
        elif (x["type"] == "route"):     # statement route,then we will implement.
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

        elif x["type"] == "yolo" or x["type"] == "region":   # Yolo is the detection layer
            anchors = x["anchors"].split(",")
            anchors = [float(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            if x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = [int(i) for i in mask]
                anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        elif x["type"] == "reorg":
            stride = int(x["stride"])
            out_channel = in_channel * (stride**2)
            img_size = int(net_info["width"])
            reorg_size = img_size / 16
            reorg = ReorgLayer(stride, in_channel, reorg_size)
            module.add_module("reorg_{0}".format(index), reorg)

        else:
            print("Something I dunno")
            assert False
        module_list.append(module)
        in_channel = out_channel
        output_channels.append(out_channel)
    return (net_info, module_list)
