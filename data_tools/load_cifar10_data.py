# coding:utf-8
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch


class LoadClassifyDataSets(Dataset):
    def __init__(self, path):
        self.path = path
        self.ids = []
        self.images = []
        for classify in os.listdir(self.path):
            temp = os.listdir(os.path.join(self.path, classify))
            self.images += temp
            self.ids += [int(classify)] * len(temp)

    def __getitem__(self, index):
        label = self.ids[index]
        img_name = self.images[index]
        img = cv2.imread(os.path.join(self.path, str(label), img_name))
        return img, label

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        imgs.append(torch.as_tensor(sample[0]))
        targets.append(torch.as_tensor(sample[1]).float())
    return torch.stack(imgs, 0), torch.stack(targets, 0)


if __name__ == '__main__':
    data_path = '/home/lintaowx/datasets/cifar10/train'
    data_set = LoadClassifyDataSets(data_path)
    batch_size = 32
    batch_iter = iter(DataLoader(data_set, batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn))
    for i in range(1000):
        images, targets = next(batch_iter)
        print(len(images))
        print(len(targets))
        print("===========================")