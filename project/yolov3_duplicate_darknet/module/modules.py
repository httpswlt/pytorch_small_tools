# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')
from utils.utils import bbox_wh_iou, bbox_iou
import pdb


class YOLOLayer(nn.Module):
    def __init__(self, anchors, ANCHORS, mask, num_classes, image_size, ignore_thres_first_loss):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.ANCHORS = ANCHORS # Anchors of all yolo layers.
        self.mask = mask
        self.num_classes = num_classes
        self.img_width = int(image_size[0])
        self.img_height = int(image_size[1])
        self.ignore_threshold = 0.5
        self.ignore_thres_first_loss = ignore_thres_first_loss # ignore threshold for first loss
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
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
        if inputs[1] is None:
            target = None
        else:
            target = inputs[1].clone()
            target = target[torch.sum(target[:, 1:6], 1) != 0]
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

        # Init loss
        loss_x, loss_y, loss_w, loss_h, \
            loss_conf_obj, loss_conf_noobj, \
            loss_cls = self.FloatTensor(1).fill_(0), self.FloatTensor(1).fill_(0), self.FloatTensor(1).fill_(0), \
            self.FloatTensor(1).fill_(0), self.FloatTensor(1).fill_(0), self.FloatTensor(1).fill_(0), \
            self.FloatTensor(1).fill_(0)

        if target is None or len(target) == 0:
            loss_conf = loss_conf_obj + loss_conf_noobj
            # total losses
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            loss_xy = loss_x + loss_y
            loss_wh = loss_w + loss_h
            return output, total_loss, loss_xy, loss_wh, loss_conf, loss_cls
        else:
            # get ignore_mask of first loss in darknet 
            ignore_mask = self._first_loss(self.prediction, target)
            # build target
            obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls, scale = self.__build_target(target)
            # merge ignore_mask and obj_mask
            ignore_mask[obj_mask == 1] = 1
            # merge ignore_mask and noobj_mask
            noobj_mask[ignore_mask == 0] = 0

            if torch.sum(obj_mask) > 0:
                # Loss:
                # x, y, w, h losses
                loss_x = torch.unsqueeze(torch.sum(
                    self.mse_loss(self.center_x[obj_mask], tx[obj_mask]) * scale[obj_mask] * scale[obj_mask]), 0)
                loss_y = torch.unsqueeze(torch.sum(
                    self.mse_loss(self.center_y[obj_mask], ty[obj_mask]) * scale[obj_mask] * scale[obj_mask]), 0)
                loss_w = torch.unsqueeze(torch.sum(
                    self.mse_loss(self.w[obj_mask], tw[obj_mask]) * scale[obj_mask] * scale[obj_mask]), 0)
                loss_h = torch.unsqueeze(torch.sum(
                    self.mse_loss(self.h[obj_mask], th[obj_mask]) * scale[obj_mask] * scale[obj_mask]), 0)
                # confidence losses
                loss_conf_obj = torch.unsqueeze(torch.sum(self.bce_loss(self.pred_conf[obj_mask], tconf[obj_mask])), 0)
                # classify losses
                loss_cls = torch.unsqueeze(torch.sum(self.bce_loss(self.pred_cls[obj_mask], tcls[obj_mask])), 0)

            if torch.sum(noobj_mask) > 0:
                loss_conf_noobj = torch.unsqueeze(torch.sum(
                    self.bce_loss(self.pred_conf[noobj_mask], tconf[noobj_mask])), 0)

            loss_conf = loss_conf_obj + loss_conf_noobj

            # total losses
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            loss_xy = loss_x + loss_y
            loss_wh = loss_w + loss_h
            # print('lxy: %f, lwh: %f, lconf: %f, lcls: %f' % (loss_xy, loss_wh, loss_conf, loss_cls))
            # print('lw: %f, lh: %f' % (loss_w, loss_h))
            return output, total_loss, loss_xy, loss_wh, loss_conf, loss_cls

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
        scale = self.FloatTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(1)

        # map target coordinate to grid size
        target_bbox = target[..., ::] * 1
        gwh_iou = target_bbox[..., 3:-1].clone()    # For best iou choose.
        target_bbox[..., 1:-1:2] *= self.grid_w
        target_bbox[..., 2:-1:2] *= self.grid_h
        gxy = target_bbox[:, 1:3]
        gwh = target_bbox[..., 3:-1]
        img_num_ = target_bbox[:, 0].long()
        img_num_ = img_num_ - img_num_[0]
        target_label = target_bbox[:, -1].long()
        
        # compute iou
        ious = torch.stack(tuple([bbox_wh_iou(anchor, gwh_iou)
                                  for anchor in torch.div(self.FloatTensor(self.ANCHORS),
                                                          self.FloatTensor((self.img_width, self.img_height)))]), 0)
        best_ious, best_index = ious.max(0)
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi_, gj_ = gxy.long().t()
        # Init scale weight 2- truth.x * truth.y
        gw_iou, gh_iou = gwh_iou.t()
        sc = 2 - gw_iou * gh_iou

        # filter anchors only to retain anchors of current yolo layer
        best_index_filter = self.ByteTensor(best_index.size()[0]).fill_(0)
        for m in self.LongTensor(self.mask):
            best_index_filter += ((best_index - m) == 0)
        # exclude outliers because of data augmentation
        outlier_filter_std = (gi_ < self.grid_w).long() + (gj_ < self.grid_h).long() + (gi_ >= 0).long() + (gj_ >= 0).long() + best_index_filter.long()
        best_index_filter = (outlier_filter_std == 5)

        img_num, best_index, gi, gj = img_num_[best_index_filter], best_index[best_index_filter], gi_[best_index_filter], gj_[best_index_filter]
        gx, gy = gx[best_index_filter], gy[best_index_filter]
        gw, gh = gw[best_index_filter], gh[best_index_filter]
        target_label = target_label[best_index_filter]
        sc = sc[best_index_filter]
        best_index -= self.mask[0] # match anchors index for current yolo layer

        if len(best_index):
            # Set masks
            obj_mask[img_num, best_index, gj, gi] = 1
            noobj_mask[img_num, best_index, gj, gi] = 0

            # Coordinates
            tx[img_num, best_index, gj, gi] = gx - gx.floor()
            ty[img_num, best_index, gj, gi] = gy - gy.floor()

            # Width and height
            tw[img_num, best_index, gj, gi] = torch.log(gw * self.stride_w / self.FloatTensor(self.anchors)[best_index][:, 0] + 1e-16)
            th[img_num, best_index, gj, gi] = torch.log(gh * self.stride_h / self.FloatTensor(self.anchors)[best_index][:, 1] + 1e-16)
            # One-hot encoding of label
            tcls[img_num, best_index, gj, gi, target_label] = 1

            # object
            tconf[img_num, best_index, gj, gi] = 1

            # Scale weight
            scale[img_num, best_index, gj, gi] = sc

        # Set noobj mask to zero where iou exceeds ignore threshold(paper said, but darknet doesn't have)
        #for i, anchor_ious in enumerate(ious[self.mask, :].t()):
        #    noobj_mask[img_num_[i], anchor_ious > self.ignore_threshold, gj_[i], gi_[i]] = 0

        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls, scale
    
    def _first_loss(self, pred, target):
        """
        
        :param pred: type: tensor: tensor.size([image_num, anchor_num, grid_j, gird_i, 5+class_num])
        :param target: type: list: [[image_num, x, y, w, h, cls],...]
        :return: ignore_mask which ignores iou(pred, truth)  > ignore_thres_first_loss
        """
        
        # Init ignore_mask which ignores iou(pred, truth)  > ignore_thres_first_loss
        ignore_mask = self.ByteTensor(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(1)

        if len(target):
            index_start = target[0][0]
            for i, pi0 in enumerate(pred):
                t = target[target[..., 0] == (i + index_start)]   # Targets for image j of batchA
                if len(t):
                    p_boxes = torch.zeros_like(pi0)
                    # transform pred to yolo box 
                    p_boxes[..., 0] = (torch.sigmoid(pi0[..., 0]) + self.grid_x[i]) / self.grid_w
                    p_boxes[..., 1] = (torch.sigmoid(pi0[..., 1]) + self.grid_y[i]) / self.grid_h
                    p_boxes[..., 2] = (torch.exp(pi0[..., 2]) * self.anchor_w[i]) / self.grid_w
                    p_boxes[..., 3] = (torch.exp(pi0[..., 3]) * self.anchor_h[i]) / self.grid_h
                    p_boxes = p_boxes.view(pi0.size()[0] * pi0.size()[1] * pi0.size()[2], 6)
            
                    # compute iou for each pred gird and all targets.
                    ious = torch.stack(tuple([bbox_iou(x, p_boxes[:, :4], False) for x in t[:, 1:5]]))
                    best_ious, best_index = ious.max(0)
                    best_ious, best_index = best_ious.view(pi0.size()[0], pi0.size()[1], pi0.size()[2], 1), \
                                            best_index.view(pi0.size()[0], pi0.size()[1], pi0.size()[2], 1)
                    ignore_mask[i][torch.squeeze(best_ious > self.ignore_thres_first_loss, 3)] = 0
        
        return ignore_mask

    def __detect(self):

        self.grid_x = torch.linspace(0, self.grid_w - 1, self.grid_w).repeat((self.grid_h, 1)).repeat(
            (self.batch_size * self.num_anchors, 1, 1)).view(self.center_x.shape).type(self.FloatTensor)
        self.grid_y = torch.linspace(0, self.grid_h - 1, self.grid_h).repeat((self.grid_w, 1)).permute(1, 0).repeat(
            (self.batch_size * self.num_anchors, 1, 1)).view(self.center_y.shape).type(self.FloatTensor)
        anchor_w = self.FloatTensor(self.scale_anchors).index_select(1, self.LongTensor([0]))
        anchor_h = self.FloatTensor(self.scale_anchors).index_select(1, self.LongTensor([1]))
        self.anchor_w = anchor_w.repeat(self.batch_size, 1).repeat(1, 1, self.grid_h * self.grid_w).view(self.w.shape)
        self.anchor_h = anchor_h.repeat(self.batch_size, 1).repeat(1, 1, self.grid_h * self.grid_w).view(self.h.shape)

# add offset
        pred_boxes = self.FloatTensor(self.prediction[..., :4].shape)
        pred_boxes[..., 0] = self.center_x.data + self.grid_x
        pred_boxes[..., 1] = self.center_y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(self.w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(self.h.data) * self.anchor_h

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
