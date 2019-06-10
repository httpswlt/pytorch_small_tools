# coding:utf-8
import torch
import torch.nn as nn
import numpy as np
import gc
import sys
from parse_model import parse_cfg, create_modules
sys.path.insert(0, '..')


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)  # all blocks from cfg file
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        self.detections = []

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            target = inputs[1]
        else:
            target = None
        self.detections = []
        block_list = self.blocks[1:]  # remove net info layer.
        outputs = {}  # We cache the outputs for the route layer
        loss = 0
        for i, block in enumerate(block_list):
            module_type = (block["type"])
            if module_type in ["convolutional", "upsample", "maxpool"]:
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
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == 'yolo':
                # x, layer_loss = self.module_list[i](x, target)
                x, layer_loss = self.module_list[i]((x, target))
                loss += layer_loss
                self.detections.append(x)
                outputs[i] = outputs[i - 1]

        return self.detections if target is None else (self.detections, loss)

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest of the values are the weights
        ptr = 0
        for i, model in enumerate(self.module_list):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional" or module_type == "deconv":
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
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
        gc.collect()
        torch.cuda.empty_cache()


def main():
    from data.load_data import *
    cfg_path = '../config/yolov3.cfg'
    data_path = '/home/lintaowx/datasets/VOC/VOCdevkit/VOC2007'
    data_set = LoadDataSets(data_path, 'trainval', AnnotationTransform(), PreProcess())
    batch_size = 12
    batch_iter = iter(DataLoader(data_set, batch_size, shuffle=False, num_workers=1, collate_fn=detection_collate))
    darknet = Darknet(cfg_path).cuda()
    darknet.train()
    for x, y in batch_iter:
        x = x.cuda()
        y = y.cuda()
        # with torch.no_grad():
        darknet(x, y)
        # torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
