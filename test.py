# coding:utf-8
import numpy as np
from backbone.vgg import vgg16
import torch


if __name__ == '__main__':
    vgg16 = vgg16(num_classes=0)
    x = np.random.normal(0, 1, size=(32, 3, 300, 300)).astype(np.float32)
    output = vgg16.forward(torch.from_numpy(x).cuda())
    pass
