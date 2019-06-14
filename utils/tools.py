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
    return y[labels.long()]


def meshgrid(x, y, row_major=True):
    """Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    """
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view((-1, 1))
    yy = b.view((-1, 1)).repeat((1, x)).view((-1, 1))
    return torch.cat((xx, yy), 1) if row_major else torch.cat((yy, xx), 1)


def change_box_order(boxes, order):
    """Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return torch.cat(((a+b)/2, b-a+1), 1)
    return torch.cat((a-b/2, a+b/2), 1)


def box_iou(box1, box2, order='xyxy'):
    """Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    # N = box1.size(0)
    # M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2]-box1[:, 0]+1) * (box1[:, 3]-box1[:, 1]+1)  # [N,]
    area2 = (box2[:, 2]-box2[:, 0]+1) * (box2[:, 3]-box2[:, 1]+1)  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou
