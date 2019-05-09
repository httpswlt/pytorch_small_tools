# coding:utf-8
import torch
from torch import nn
import numpy as np
import math


class YOlOLoss(nn.Module):
    def __init__(self, classes, anchors, img_size):
        super(YOlOLoss, self).__init__()
        self.img_size = img_size
        self.anchors = anchors
        self.classes = classes
        self.bbox_attrs = 5 + classes
        self.ignore_threshold = 0.5
        self.batch_size = 1
        self.num_anchors = len(self.anchors)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.grid_size = 0

    def forward(self, pred, targets):
        batch_size = pred.size(0)
        self.batch_size = batch_size
        pred_im_h = pred.size(2)
        pred_im_w = pred.size(3)
        stride_h = self.img_size[0] / pred_im_h
        stride_w = self.img_size[1] / pred_im_w

        scale_anchors = [(w / stride_w, h / stride_h) for w, h in self.anchors]

        prediction = pred.view(batch_size, self.num_anchors, self.bbox_attrs,
                               pred_im_h, pred_im_w).permute(0, 1, 3, 4, 2).contiguous()

        # get outputs
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])


        if targets is not None:
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = \
                self.build_targets(targets, scale_anchors, pred_im_w, pred_im_h, self.ignore_threshold)
            #  losses.
            loss_x = self.mse_loss(x * mask, tx * mask)
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * mask, mask) + \
                        0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)


    def build_targets(self, target, anchors, w, h, ignore_threshold):
        from utils.box_utils import jaccard
        obj_mask = torch.zeros(self.batch_size, self.num_anchors, h, w)
        noobj_mask = torch.ones(self.batch_size, self.num_anchors, h, w)
        tx = torch.zeros(self.batch_size, self.num_anchors, h, w)
        ty = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tw = torch.zeros(self.batch_size, self.num_anchors, h, w)
        th = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tconf = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tcls = torch.zeros(self.batch_size, self.num_anchors, h, w, self.classes)

        target[:, :, :-1:2] *= w
        target[:, :, 1:-1:2] *= w




        # for bs in range(self.batch_size):
        #     for t in range(target.shape[1]):
        #         if target[bs, t].sum() == 0:
        #             continue
        #         # Convert to position relative to box
        #         gx = target[bs, t, 1] * w
        #         gy = target[bs, t, 2] * h
        #         gw = target[bs, t, 3] * w
        #         gh = target[bs, t, 4] * h
        #         # get grid box indices
        #         gi = int(gx)
        #         gj = int(gy)
        #         # get shape of gt box
        #         gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
        #         # get shape of anchor box
        #         anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
        #                                                           np.array(anchors)), 1))
        #         # Calculate iou between gt and anchor shapes
        #         anch_ious = jaccard(gt_box, anchor_shapes).squeeze()
        #         # Where the overlap is larger than threshold set mask to zero (ignore)
        #         noobj_mask[bs, anch_ious > ignore_threshold, gj, gi] = 0
        #         # Find the best matching anchor box
        #         best_n = np.argmax(anch_ious)
        #
        #         # Masks
        #         obj_mask[bs, best_n, gj, gi] = 1
        #         # Coordinates
        #         tx[bs, best_n, gj, gi] = gx - gi
        #         ty[bs, best_n, gj, gi] = gy - gj
        #         # Width and height
        #         tw[bs, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
        #         th[bs, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
        #         # object
        #         tconf[bs, best_n, gj, gi] = 1
        #         # One-hot encoding of label
        #         tcls[bs, best_n, gj, gi, int(target[bs, t, 0])] = 1

            # return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls

    def mse_loss(self):
        pass


def main():
    from models.yolov3 import YOLOV3, parameter
    model = YOLOV3(parameter)
    x = torch.randn((1, 3, 416, 416))
    y0, y1, y2 = model(x)
    target = torch.rand((1, 3, 5))
    loss = YOlOLoss(21, parameter['yolo']['anchors'][0], (416, 416))
    loss(y0, target)


if __name__ == '__main__':
    main()
