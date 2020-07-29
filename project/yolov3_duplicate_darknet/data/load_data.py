# coding:utf-8
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import torch
from data_utils import AnnotationTransform, PreProcess


class LoadDataSets(Dataset):
    def __init__(self, data_path, image='trainval', target_transform=None, pre_process=None):
        """

        :param data_path: dataset path(should contain Annotations, JPEGImages and so on.)
        :param image:
        :param target_transform:
        :param pre_process:
        """
        self.data_path = data_path
        self.target_transform = target_transform
        self.pre_process = pre_process
        self._anno_path = os.path.join(self.data_path, 'Annotations', '%s.xml')
        self._img_path = os.path.join(self.data_path, 'JPEGImages', '%s.jpg')
        self.ids = []
        self.target = None
        self.img = None
        #for line in open(os.path.join(self.data_path, 'ImageSets', 'Main', image+'.txt')):
        for line in open(image):

            self.ids.append((line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        self.img = cv2.imread(self._img_path % img_id, cv2.IMREAD_COLOR)

        if self.target_transform is not None:
            self.target = self.target_transform(self._anno_path % img_id)

        if not (isinstance(self.target, np.ndarray) and isinstance(self.img, np.ndarray)):
            print("Need Target And Image For Object Detection.")
            exit(0)

        if self.pre_process is not None:
            self.img,  self.target = self.pre_process(self.img, self.target)
        return self.img, self.target

    def __len__(self):
        return len(self.ids)


def detection_collate(batch):
    """

    :param batch:
    :return:    images:     type: tensor, shape:(batch_size, channel, w, h))
                targets:    type: list, len(targets) equal batch_size.
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        imgs.append(sample[0])
        temp = torch.from_numpy(sample[1]).float()
        targets.append(torch.cat((torch.Tensor([[_]] * temp.size(0)), temp), 1))
    return torch.stack(imgs, 0), torch.cat(targets, 0)


if __name__ == '__main__':
    # data_path = '/mnt/storage/project/data/VOCdevkit/VOC2007'
    data_path = '~/datasets/VOC/VOCdevkit/VOC2007'
    data_set = LoadDataSets(data_path, 'train', AnnotationTransform(), PreProcess())
    batch_size = 32
    batch_iter = iter(DataLoader(data_set, batch_size, shuffle=False, num_workers=1, collate_fn=detection_collate))
    for i in range(1000):
        images, targets = next(batch_iter)
        print(len(images))
        print(len(targets))
        print("===========================")
