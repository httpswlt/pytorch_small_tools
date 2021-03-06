# coding:utf-8
from torch.nn import functional as F


def ic_layer(x, p):
    """
    Inference:
        https://arxiv.org/pdf/1905.05928.pdf
        paper recommend use: ReLU-IC-Conv2D
    :param x: input data
    :param p: random to drop probability
    :return:
    """
    x = F.batch_norm(x)
    x = F.dropout(x, p)
    return x
