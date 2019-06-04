# coding:utf-8
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import random

class PreProcess(object):
    """
        function:
            1. resize image and reconstitution bbox coordinate.
            2. reduce mean value.
    """
    def __init__(self, resize=(416, 416), rgb_means=(104, 117, 123)):
        self.means = rgb_means
        self.resize = resize

    def __call__(self, image, target):
        boxes = target[:, :-1].copy()
        if len(boxes) == 0:
            targets = np.zeros((1, 5))
            image, _ = self.resize_mean(image, targets, self.resize, self.means)
            return torch.from_numpy(image), targets
        img, tar = self.resize_mean(image.copy(), target.copy(), self.resize, self.means)
        return torch.from_numpy(img), tar

    def resize_mean(self, image, target, im_size, mean):
        h, w, _ = image.shape
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image = cv2.resize(image, im_size, interpolation=interp_method)
        image = image.astype(np.float32)
        image -= mean
        h_, w_, _ = image.shape
        target[:, 0:-1:2] *= (float(w_) / w)
        target[:, 1:-1:2] *= (float(h_) / h)

        # x1, y2, x2, y2 normalization
        target[:, 0:-1:2] /= float(w_)
        target[:, 1:-1:2] /= float(h_)

        # convert x1,y1,x2,y2 to x,y,w,h
        target = self.to_xywh(target)

        return image.transpose(2, 0, 1), target

    @staticmethod
    def to_xywh(target):
        x1 = target[:, 0].reshape(-1, 1)
        y1 = target[:, 1].reshape(-1, 1)
        x2 = target[:, 2].reshape(-1, 1)
        y2 = target[:, 3].reshape(-1, 1)
        cls = target[:, -1].reshape(-1, 1)
        w = (x2 - x1)
        h = (y2 - y1)
        return np.hstack((x1, y1, w, h, cls))


class AnnotationTransform(object):
    """
            parse xml file, acquire bbox coordinate and classify.

    """
    VOC_CLASSES = ('__background__',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, class_to_ind=None, keep_difficult=True):
        """

        :param class_to_ind:    format : {__background__': 0, 'aeroplane': 1, ...}
        :param keep_difficult:  whether keep some difficult annotation object
        """
        self.class_to_ind = class_to_ind or dict(
            zip(self.VOC_CLASSES, range(len(self.VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, path):
        """
        :param path:  the path of xml object that need to parse
        :return:
        """
        target = ET.parse(path).getroot()
        res = np.empty((0, 5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]



