# coding:utf-8
import torch
import numpy as np


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
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

def predict_transform(prediction, im_dim, anchors, num_classes, gpu_num):
    batch_size = prediction.size(0)
    stride_w = im_dim[0] // prediction.size(3)
    stride_h = im_dim[1] // prediction.size(2)
    grid_size_w = im_dim[0] // stride_w
    grid_size_h = im_dim[1] // stride_h
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    anchors = [(a[0] / stride_w, a[1] / stride_h) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size_w * grid_size_h)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size_w * grid_size_h * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    fp16 = True if '16' in str(prediction.dtype) else False
    # Add the center offsets
    grid_len_w = np.arange(grid_size_w, dtype=np.int32)
    grid_len_h = np.arange(grid_size_h, dtype=np.int32)
    a, b = np.meshgrid(grid_len_w, grid_len_h)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if fp16:
        x_offset = x_offset.half()
        y_offset = y_offset.half()

    if gpu_num != None:
        x_offset = x_offset.cuda(device=gpu_num)
        y_offset = y_offset.cuda(device=gpu_num)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if fp16:
        anchors = anchors.half()

    if gpu_num != None:
        anchors = anchors.cuda(device=gpu_num)

    anchors = anchors.repeat(grid_size_w * grid_size_h, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Softmax the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, 0] *= stride_w
    prediction[:, :, 2] *= stride_w
    prediction[:, :, 1] *= stride_h
    prediction[:, :, 3] *= stride_h

    return prediction
    

def region_predict_transform(prediction, im_dim, anchors, num_classes, gpu_num):
    batch_size = prediction.size(0)
    grid_size_w = prediction.size(3)
    grid_size_h = prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size_w * grid_size_h)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size_w * grid_size_h * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid_len_w = np.arange(grid_size_w, dtype=np.int32)
    grid_len_h = np.arange(grid_size_h, dtype=np.int32)
    a, b = np.meshgrid(grid_len_w, grid_len_h)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if gpu_num != None:
        x_offset = x_offset.cuda(device=gpu_num)
        y_offset = y_offset.cuda(device=gpu_num)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if gpu_num != None:
        anchors = anchors.cuda(device=gpu_num)

    anchors = anchors.repeat(grid_size_w * grid_size_h, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Softmax the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.nn.functional.softmax(prediction[:, :, 5: 5 + num_classes] ,dim=2)
    	
    prediction[:, :, 0] *= float(im_dim[0])/float(grid_size_w)
    prediction[:, :, 2] *= float(im_dim[0])/float(grid_size_w)
    prediction[:, :, 1] *= float(im_dim[1])/float(grid_size_h)
    prediction[:, :, 3] *= float(im_dim[1])/float(grid_size_h)

    return prediction
    
    
def reorg(x, stride, gpu_num):
    input = x.data
    B,C,H,W = input.size()
    tmp = input.view(B, C, H/stride, stride, W/stride, stride).transpose(3,4).contiguous()
    tmp = tmp.view(B, C, H/stride*W/stride, stride*stride).transpose(2,3).contiguous()
    tmp = tmp.view(B, C, stride*stride, H/stride, W/stride).transpose(1,2).contiguous()
    
    tmp = tmp.view(B, stride*C*H, W/stride)
    new_index = np.arange(stride*C*H)
    for i in range(stride*C*H):
        if i % 2 == 0:
            if i < stride*C*H / 2:
                new_index[i+1] = new_index[i] + stride*C*H / 2
            else:
                new_index[i] = i - stride*C*H / 2 + 1
                new_index[i+1] = new_index[i] + stride*C*H / 2

    tmp[0] = tmp[0][new_index]
    result = tmp.view(B, stride*stride*C, H/stride, W/stride)
    
    return result 
    
