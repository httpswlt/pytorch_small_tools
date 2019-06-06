# coding:utf-8
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, loc_preds, cls_preds, targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
            Args:
              loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
              cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
              targets: (tensor) encoded target labels, sized [batch_size, #anchors].

            loss:
              (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """