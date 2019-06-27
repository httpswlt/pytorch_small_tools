# coding:utf-8
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import cv2
import torch
import random
import math


class YoloDataSets(Dataset):
    debI = 0

    def __init__(self, data_path, input_size=(1280, 768), batch_size=4, image_set='trainval', jitter_x=0.3,
                 jitter_y=0.3, augment=True, hue=0.1, saturation=1.5, exposure=1.5, angle=0):
        """

        :param data_path: dataset path(should contain Annotations, JPEGImages and so on.)
        :param image:
        :param target_transform:
        :param pre_process:
        """
        self.input_size_width = input_size[0]
        self.input_size_height = input_size[1]
        self.data_path = data_path
        #self.data_set_path = os.path.join(self.data_path, 'ImageSets', 'Main', image_set + '.txt')
        self.data_set_path = image_set
        self._anno_path = os.path.join(self.data_path, 'Annotations')
        self._img_path = os.path.join(self.data_path, 'JPEGImages')
        self.batch_count = 0
        self.augment = augment
        with open(self.data_set_path, 'r') as f:
            img_files = f.read().splitlines()
            img_files = list(filter(lambda x: len(x) > 0, img_files))
            self.img_files = [os.path.join(self._img_path, ix+'.jpg') for ix in img_files]
        n = len(self.img_files)
        self.ids = img_files
        self.jitter_x = jitter_x
        self.jitter_y = jitter_y
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.label_files = [x.replace('JPEGImages', 'labels').
                                replace('.jpeg', '.txt').
                                replace('.jpg', '.txt').
                                replace('.bmp', '.txt').
                                replace('.png', '.txt') for x in self.img_files]

        self.labels = [np.zeros((0, 5))] * n
        self.labels_pre = [np.zeros((0, 5))] * n


        iter = tqdm(self.label_files, desc='Reading labels') if n > 1000 else self.label_files
        for i, file in enumerate(iter):
            try:
                with open(file, 'r') as f:
                    self.labels_pre[i] = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                with open(file, 'r') as f:
                    self.labels[i] = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                self.labels[i][:, :4] = self.labels_pre[i][:, 1:5]
                self.labels[i][:, 4] = self.labels_pre[i][:, 0]

            except:
                print("Missing " + file)
                pass  # missing label file

    
    @staticmethod
    def constrain(min, max, value):
        if value > max:
            return max
        if value < min:
            return min
        return value

    @staticmethod
    def rand():
        return int(random.random() * 32767)

    def __getitem__(self, index):
        img_id = self.img_files[index]
        img_name = img_id.split('/')[-1]
        img = None
        img0 = cv2.imread(img_id)
        target = self.labels[index]

        img_height, img_width, img_channel = img0.shape

        img = cv2.resize(img0, (int(self.input_size_width), int(self.input_size_height)))
        new_target = target
        valid_count = len(target)


        labels_out = torch.zeros((130, 6))

        if valid_count:
            labels_out[:valid_count, 1:] = torch.from_numpy(new_target[:valid_count])

        img = torch.from_numpy(img).float()
        img = img.permute((2, 0, 1))
        img = torch.unsqueeze(img.div(255.0), 0)

        img0 = torch.from_numpy(img0).float()

        return img_name, img, img0, labels_out

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        """

        :param batch:
        :return:    images:     type: tensor, shape:(batch_size, channel, w, h))
                    targets:    type: list, len(targets) equal batch_size.
        """
        img_names, imgs, img0, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        self.batch_count += 1

        return img_names, torch.squeeze(torch.stack(imgs, 0), 1), torch.squeeze(torch.stack(img0, 0), 1), torch.cat(targets, 0)



if __name__ == '__main__':
    data_path = '/home/lingc1/data/sports-training-data/player_detection/training_dataset_debug'
    data_set = YoloDataSets(data_path, image_set='train_freed_2k')
    batch_size = 2
    dataloader = DataLoader(data_set, batch_size, shuffle=False, num_workers=1, collate_fn=data_set.collate_fn)
    for ii in range(2):
        for i, (imgs, targets) in enumerate(dataloader):
            print(len(imgs))
            print(len(targets))
            print("===========================")

    # for i in range(1000):
    #     images, targets = next(batch_iter)
    #     print(len(images))
    #     print(len(targets))
    #     print("===========================")
