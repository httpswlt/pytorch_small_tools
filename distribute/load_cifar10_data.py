# coding:utf-8
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch


class LoadClassifyDataSets(Dataset):
    def __init__(self, path, image_size):
        self.path = path
        self.ids = []
        self.images = []
        self.images_size = image_size
        for classify in os.listdir(self.path):
            temp = os.listdir(os.path.join(self.path, classify))
            self.images += temp
            self.ids += [int(classify)] * len(temp)

    def __getitem__(self, index):
        label = self.ids[index]
        img_name = self.images[index]
        img = cv2.imread(os.path.join(self.path, str(label), img_name))
        if img.shape[0] != self.images_size:
            img = cv2.resize(img, (self.images_size, self.images_size))

        return img.transpose(2, 0, 1), label

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        imgs.append(torch.as_tensor(sample[0]).float())
        targets.append(torch.as_tensor(sample[1]).long())
    return torch.stack(imgs, 0), torch.stack(targets, 0)


if __name__ == '__main__':
    data_path = '~/datasets/cifar10/train'
    data_set = LoadClassifyDataSets(data_path, 227)
    batch_size = 64
    loader = DataLoader(data_set, batch_size, shuffle=True, num_workers=1, drop_last=True, collate_fn=collate_fn)

    for i in range(10):
        temp = iter(loader)
        images, targets = next(temp)
        while images is not None:
            print(len(images))
            print(len(targets))
            print("===========================")
            images, targets = next(temp, (0, 0))
            if isinstance(images, int):
                break
