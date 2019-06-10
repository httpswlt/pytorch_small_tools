# coding:utf-8
import math
import torch
from utils.tools import meshgrid, change_box_order, box_iou


class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32 * 32., 64 * 64., 128 * 128., 256 * 256., 512 * 512.]  # p3 -> p7
        self.aspect_ratios = [1 / 2., 1 / 1., 2 / 1.]
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        """Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].

        :return:
        """
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:   # w/h = ar
                h = math.sqrt(s / ar)
                w = ar * h
                for sr in self.scale_ratios:
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view((num_fms, -1, 2))

    def encode(self, targets, input_size):
        boxes = targets[..., :4]
        labels = targets[..., -1]
        input_size = torch.Tensor([input_size, input_size]) if isinstance(input_size, int) \
            else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = box_iou(anchor_boxes, boxes, order='xywh')

        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]
        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious < 0.5] = 0
        ignore = (max_ious > 0.4) & (max_ious < 0.5)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1  # for now just mark ignored to -1
        return loc_targets, cls_targets

    def _get_anchor_boxes(self, input_size):
        """Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        """
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size / pow(2., i+3)).ceil() for i in range(num_fms)]   # p3 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h).type(torch.FloatTensor)
            xy = (xy * grid_size).view((fm_h, fm_w, 1, 2)).expand(fm_h, fm_w, 9, 2)
            wh = self.anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)
            box = torch.cat((xy, wh), 3)  # [x,y,w,h]
            boxes.append(box.view((-1, 4)))
        return torch.cat(tuple(boxes), 0)

