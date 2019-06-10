# coding:utf-8
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
import random
import torch
from encoder import DataEncoder


class PreProcess(object):
    """
        function:
            1. resize image and reconstitution bbox coordinate.
            2. reduce mean value.
    """
    def __init__(self, resize=(300, 300), rgb_means=(104, 117, 123)):
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

        # # x1, y2, x2, y2 normalization
        # target[:, 0:-1:2] /= float(w_)
        # target[:, 1:-1:2] /= float(h_)
        #
        # # convert x1,y1,x2,y2 to x,y,w,h
        # target = self.to_xywh(target)

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
        self.class_to_ind = class_to_ind or dict(
            zip(self.VOC_CLASSES, range(len(self.VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
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


class LoadVocDataSets(Dataset):
    """
        Generate inference data.
    """
    def __init__(self, data_path, image, target_transform=None, pre_process=None):
        self.data_path = data_path
        self.target_transform = target_transform
        self.pre_process = pre_process
        self._anno_path = os.path.join(self.data_path, 'Annotations', '%s.xml')
        self._img_path = os.path.join(self.data_path, 'JPEGImages', '%s.jpg')
        self.ids = []
        self.encoder = DataEncoder()
        for line in open(os.path.join(self.data_path, 'ImageSets', 'Main', image+'.txt')):
            self.ids.append((line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._anno_path % img_id).getroot()
        img = cv2.imread(self._img_path % img_id, cv2.IMREAD_COLOR)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.pre_process is not None:
            img, target = self.pre_process(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def detection_collate(self, batch):
        imgs = []
        loc_targets = []
        cls_targets = []
        for _, sample in enumerate(batch):
            imgs.append(sample[0])
            loc_target, cls_target = self.encoder.encode(torch.from_numpy(sample[1]).float(), (sample[0].shape[2], sample[0].shape[1]))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return torch.stack(imgs, 0), torch.stack(loc_targets), torch.stack(cls_targets)


if __name__ == '__main__':
    # data_path = '/mnt/storage/project/data/VOCdevkit/VOC2007'
    data_path = '/home/lintaowx/datasets/VOC/VOCdevkit/VOC2007'
    data_set = LoadVocDataSets(data_path, 'trainval', AnnotationTransform(), PreProcess())
    batch_size = 32
    batch_iter = iter(DataLoader(data_set, batch_size, shuffle=False, num_workers=1, collate_fn=data_set.detection_collate))
    for i in range(1000):
        images, loc_targets, cls_targets = next(batch_iter)
        print(len(images))
        print(len(loc_targets))
        print(len(cls_targets))
        print("===========================")
