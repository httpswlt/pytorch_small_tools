# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')
from utils.utils import bbox_wh_iou


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, image_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.img_width = int(image_size[0])
        self.img_height = int(image_size[1])
        self.ignore_threshold = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 0.5

        # common properties
        self.batch_size = 0
        self.grid_h = 0
        self.grid_w = 0
        self.scale_anchors = []
        self.stride_h = 0
        self.stride_w = 0
        self.prediction = None
        self.center_x = 0
        self.center_y = 0
        self.w = 0
        self.h = 0
        self.pred_conf = 0
        self.pred_cls = 0
        self.ByteTensor = None
        self.FloatTensor = None
        self.LongTensor = None

    def forward(self, inputs):
        """

        :param inputs: type: tuple, (x, target)
        :return:
        """
        x = inputs[0]
        target = inputs[1]
        self.batch_size = x.size(0)
        self.grid_h = x.size(2)
        self.grid_w = x.size(3)
        self.stride_h = self.img_height / self.grid_h
        self.stride_w = self.img_width / self.grid_w

        self.scale_anchors = [(w / self.stride_w, h / self.stride_h) for w, h in self.anchors]

        self.prediction = (
                x.view(self.batch_size, self.num_anchors, self.num_classes + 5, self.grid_h, self.grid_w)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # support CUDA
        self.ByteTensor = torch.cuda.ByteTensor if self.prediction.is_cuda else torch.ByteTensor
        self.FloatTensor = torch.cuda.FloatTensor if self.prediction.is_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.prediction.is_cuda else torch.LongTensor

        # get outputs
        self.center_x = torch.sigmoid(self.prediction[..., 0])  # center x
        self.center_y = torch.sigmoid(self.prediction[..., 1])  # center y
        self.w = self.prediction[..., 2]  # predict box width
        self.h = self.prediction[..., 3]  # predict box height
        self.pred_conf = torch.sigmoid(self.prediction[..., 4])  # box confidence
        self.pred_cls = torch.sigmoid(self.prediction[..., 5:])  # object confidence of per classify

        output = self.__detect()
        if target is None:
            return output, 0
        else:
            obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.__build_target(target)
            # Loss:
            # x, y, w, h losses
            loss_x = self.mse_loss(self.center_x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(self.center_x[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(self.w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(self.w[obj_mask], th[obj_mask])
            # confidence losses
            loss_conf_obj = self.bce_loss(self.pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(self.pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # classify losses
            loss_cls = self.bce_loss(self.pred_cls[obj_mask], tcls[obj_mask])
            # total losses
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return output, total_loss

    def __build_target(self, target):
        """

        :param target: type: list: [[image_num, x, y, w, h, cls],...]
        :return:
        """
        obj_mask = self.ByteTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0)
        noobj_mask = self.ByteTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(1)
        tx = self.FloatTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0)
        ty = self.FloatTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0)
        tw = self.FloatTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0)
        th = self.FloatTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0)
        tconf = self.FloatTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0)
        tcls = self.FloatTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w, self.num_classes).fill_(0)

        # map target coordinate to grid size
        target_bbox = target[..., ::] * 1
        target_bbox[..., 1:-1:2] *= self.grid_w
        target_bbox[..., 2:-1:2] *= self.grid_h
        gxy = target_bbox[:, 1:3]
        gwh = target_bbox[..., 3:-1]
        img_num = target_bbox[:, 0].long()
        target_label = target_bbox[:, -1].long()
        # compute iou
        ious = torch.stack(tuple([bbox_wh_iou(self.FloatTensor(anchor), gwh) for anchor in self.anchors]), 0)
        best_ious, best_index = ious.max(0)
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        obj_mask[img_num, best_index, gj, gi] = 1
        noobj_mask[img_num, best_index, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[img_num[i], anchor_ious > self.ignore_threshold, gj[i], gi[i]] = 0

        # Coordinates
        tx[img_num, best_index, gj, gi] = gx - gx.floor()
        ty[img_num, best_index, gj, gi] = gy - gy.floor()

        # Width and height
        tw[img_num, best_index, gj, gi] = torch.log(gw / self.FloatTensor(self.anchors)[best_index][:, 0] + 1e-16)
        th[img_num, best_index, gj, gi] = torch.log(gh / self.FloatTensor(self.anchors)[best_index][:, 1] + 1e-16)

        # One-hot encoding of label
        tcls[img_num, best_index, gj, gi, target_label] = 1

        # object
        tconf[img_num, best_index, gj, gi] = 1

        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls

    def __detect(self):
        # compute offset
        grid_x = torch.linspace(0, self.grid_w - 1, self.grid_w).repeat((self.grid_w, 1)).repeat(
            (self.batch_size * self.num_anchors, 1, 1)).view(self.center_x.shape).type(self.FloatTensor)
        grid_y = torch.linspace(0, self.grid_h - 1, self.grid_h).repeat((self.grid_h, 1)).repeat(
            (self.batch_size * self.num_anchors, 1, 1)).view(self.center_y.shape).type(self.FloatTensor)
        anchor_w = self.FloatTensor(self.scale_anchors).index_select(1, self.LongTensor([0]))
        anchor_h = self.FloatTensor(self.scale_anchors).index_select(1, self.LongTensor([1]))
        anchor_w = anchor_w.repeat(self.batch_size, 1).repeat(1, 1, self.grid_h * self.grid_w).view(self.w.shape)
        anchor_h = anchor_h.repeat(self.batch_size, 1).repeat(1, 1, self.grid_h * self.grid_w).view(self.h.shape)

        # add offset
        pred_boxes = self.FloatTensor(self.prediction[..., :4].shape)
        pred_boxes[..., 0] = self.center_x.data + grid_x
        pred_boxes[..., 1] = self.center_y.data + grid_y
        pred_boxes[..., 2] = torch.exp(self.w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(self.h.data) * anchor_h

        _scale = self.FloatTensor([self.stride_w, self.stride_h] * 2)
        output = torch.cat(
            (
                pred_boxes.view((self.batch_size, -1, 4)) * _scale,
                self.pred_conf.view((self.batch_size, -1, 1)),
                self.pred_cls.view((self.batch_size, -1, self.num_classes)),
            ),
            -1,
        )
        return output


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    __slots__ = []

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        pass


class MaxPoolStride1(nn.Module):
    __slots__ = ['kernel_size', 'pad']

    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, [0, self.pad, 0, self.pad], mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class DetectionLayer(nn.Module):
    __slots__ = ['anchors']

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x):
        pass
