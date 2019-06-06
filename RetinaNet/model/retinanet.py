# coding:utf-8
import torch
from torch import nn
from backbone.fpn import FPN50


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=10):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().\
                view(x.size(0), -1, 4)  # [N, anchor_nums*4,H,W] -> [N,H,W, anchor_nums*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().\
                view(x.size(0), -1, self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(tuple(loc_preds), 1), torch.cat(tuple(cls_preds), 1)

    @staticmethod
    def _make_head(out_channels):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()