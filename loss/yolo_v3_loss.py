# coding:utf-8
import torch
from torch import nn
import numpy as np
import sys
from utils.box_utils import jaccard
sys.path.insert(0, '../')


class YOlOLoss(nn.Module):
    def __init__(self, conf, i):
        super(YOlOLoss, self).__init__()
        self.img_size = conf['img_size']
        self.anchors = conf['anchors'][i]
        self.classes = conf['classes']
        self.bbox_attrs = 5 + self.classes
        self.ignore_threshold = conf['ignore_threshold']
        self.batch_size = conf['batch_size']
        self.num_anchors = len(self.anchors)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = conf['obj_scale']
        self.noobj_scale = conf['noobj_scale']

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
            obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls = \
                self.build_targets(targets, scale_anchors, pred_im_w, pred_im_h, self.ignore_threshold)
            #  losses.
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return total_loss

    def build_targets(self, target, anchors, w, h, ignore_threshold):
        obj_mask = torch.zeros(self.batch_size, self.num_anchors, h, w).byte()
        noobj_mask = torch.ones(self.batch_size, self.num_anchors, h, w).byte()
        tx = torch.zeros(self.batch_size, self.num_anchors, h, w)
        ty = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tw = torch.zeros(self.batch_size, self.num_anchors, h, w)
        th = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tconf = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tcls = torch.zeros(self.batch_size, self.num_anchors, h, w, self.classes)

        target_bbox = target[..., ::] * 1
        target_bbox[..., :-1:2] *= w
        target_bbox[..., 1:-1:2] *= h
        gwh = target_bbox[..., 2:-1]
        anchor_shape = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)), np.array(anchors)), 1))
        gwh_shape = torch.FloatTensor(np.concatenate((np.zeros_like(gwh), np.array(gwh)), 2)).contiguous()
        for bs in range(target_bbox.size(0)):
            for tg in range(target_bbox.size(1)):
                if target_bbox[bs, tg][:-1].sum() == 0:
                    continue
                gx = target_bbox[bs, tg, 0]
                gy = target_bbox[bs, tg, 1]
                gw = target_bbox[bs, tg, 2]
                gh = target_bbox[bs, tg, 3]
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                gi = 1 if gi == 0 else gi
                gj = 1 if gj == 0 else gj
                # acquire the best anchor by computing ious
                anchor_ious = jaccard(gwh_shape[bs, tg].unsqueeze(0), anchor_shape).squeeze()
                # if overlap more than threshold, set no object mask zero.
                noobj_mask[bs, anchor_ious > ignore_threshold, gi, gj] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anchor_ious)
                # mask
                obj_mask[bs, best_n, gj, gi] = 1
                noobj_mask[bs, best_n, gj, gi] = 0
                # coordinate
                tx[bs, best_n, gj, gi] = gx - gi
                ty[bs, best_n, gj, gi] = gy - gj
                tw[bs, best_n, gj, gi] = torch.log(gw / anchors[best_n][0] + 1e-16)
                th[bs, best_n, gj, gi] = torch.log(gh / anchors[best_n][1] + 1e-16)
                # object
                tconf[bs, best_n, gj, gi] = 1
                # one-hot encoding of label
                tcls[bs, best_n, gj, gi, int(target_bbox[bs, tg, -1])] = 1
        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls

    @staticmethod
    def bbox_wh_iou(wh1, wh2):
        wh2 = wh2.t()
        wh1 = torch.as_tensor(wh1)
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
        return inter_area / union_area


def main():
    from models.yolov3 import YOLOV3, parameter
    model = YOLOV3(parameter)
    x = torch.randn((2, 3, 416, 416))
    y0, y1, y2 = model(x)
    target = torch.rand((2, 4, 5))
    loss = YOlOLoss(21, parameter['yolo']['anchors'][0], (416, 416))
    loss(y0, target)


if __name__ == '__main__':
    main()
