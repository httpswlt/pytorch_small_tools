# coding:utf-8
import torch
from torch import nn


def adjust_learning_rate(lr, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr_ = 1e-6 + (lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr_ = lr * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_


def init_parameters_msra(model):
    # init weight by MSRA
    init_parameters(model, nn.init.kaiming_normal_)


def init_parameters_xavier(model):
    # init weight by xavier
    init_parameters(model, nn.init.xavier_uniform_)


def init_parameters(model, meth):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            meth(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def draw_image(image, coordinate):
    import cv2
    x1, y1, x2, y2 = int(coordinate[0]), int(coordinate[1]), int(coordinate[2]), int(coordinate[3])
    cv2.putText(image, "123", (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]
