# coding:utf-8
import torch
from torch import nn
import numpy as np
import sys
from utils.box_utils import jaccard
sys.path.insert(0, '../')


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
        obj_mask = torch.zeros(self.batch_size, self.num_anchors, h, w)
        noobj_mask = torch.ones(self.batch_size, self.num_anchors, h, w)
        tx = torch.zeros(self.batch_size, self.num_anchors, h, w)
        ty = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tw = torch.zeros(self.batch_size, self.num_anchors, h, w)
        th = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tconf = torch.zeros(self.batch_size, self.num_anchors, h, w)
        tcls = torch.zeros(self.batch_size, self.num_anchors, h, w, self.classes)

        target_bbox = target[..., ::] * 1
        target_bbox[..., :-1:2] *= w
        target_bbox[..., 1:-1:2] *= h
        # gxy = target_bbox[..., :2]
        gwh = target_bbox[..., 2:-1]
        anchor_shape = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)), np.array(anchors)), 1))
        gwh_shape = torch.FloatTensor(np.concatenate((np.zeros_like(gwh), np.array(gwh)), 2)).contiguous()
        # gwh_reshape = gwh_shape.reshape(-1, 4)
        # ious = [jaccard(gwh_reshape[i, :].unsqueeze(0), anchor_shape) for i in range(gwh_reshape.size(0))]
        # ious = torch.stack(ious).contiguous().squeeze()
        # gx = gxy[:, :, 0]
        # gy = gxy[:, :, 1]
        # gi = gx.long()
        # gj = gy.long()
        # _, _, gw, gh = gwh_reshape.t()
        # gw = gw.reshape(self.batch_size, -1)
        # gh = gh.reshape(self.batch_size, -1)
        # # set object mask
        # best_iou, best_index = torch.max(ious, 1)
        # ious = ious.reshape((self.batch_size, -1, self.num_anchors))
        # best_index = best_index.reshape((self.batch_size, self.num_anchors))
        # for i in range(self.batch_size):
        #     temp_i = gi[i, ...]
        #     temp_j = gj[i, ...]
        #     best_n = best_index[i, ...]
        #     obj_mask[i, best_n, temp_j, temp_i] = 1
        #     noobj_mask[i, best_n, temp_j, temp_i] = 0
        #     noobj_mask[i, ious[i:, ] > self.ignore_threshold, temp_j, temp_i] = 0
        #     tx[i, best_n, temp_j, temp_i] = gx[i,  ...] - gx[i, ...].floor()
        #     ty[i, best_n, temp_j, temp_i] = gy[i,  ...] - gy[i, ...].floor()
        #     tw[i, best_n, temp_j, temp_i] = torch.log(gw[i, ...] / anchors[best_n][:, 0] + 1e-16)
        #     th[i, best_n, temp_j, temp_i] = torch.log(gh[i, ...] / anchors[best_n][:, 1] + 1e-16)
        #
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






        # return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls


        # ios = torch.stack(ious, 0).reshape((target_bbox.size(0), -1, anchor_shape.size(0)))
        # temp = gxy.int()
        # gi = temp[..., 0]
        # gj = temp[..., 1]
        # obj_mask
        pass


        # def construct_bbox(wh):
        #     temp = wh.reshape(-1, 2)
        #     return torch.cat((torch.zeros_like(temp), temp), 1)
        # gwh_box = construct_bbox(gwh)
        # a = [jaccard(gwh_box, construct_bbox(torch.as_tensor(anchor))) for anchor in anchors]
        #
        # pass
        # jaccard(anchor_shape, torch.FloatTensor(np.array([0, 0, gw, gh])))

        # for bs in range(self.batch_size):
        #     for t in range(target.shape[1]):
        #         if target[bs, t].sum() == 0:
        #             continue
        #         # Convert to position relative to box
        #         gx = target[bs, t, 1] * w
        #         gy = target[bs, t, 2] * h
        #         gw = target[bs, t, 3] * w
        #         gh = target[bs, t, 4] * h
        #         # get grid box index, then compute coordinate offset.
        #         gi = int(gx)
        #         gj = int(gy)
        #         # construct gt box by wh
        #         gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
        #         # construct default box by default w,h of anchor
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
        #
        #     return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls

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
