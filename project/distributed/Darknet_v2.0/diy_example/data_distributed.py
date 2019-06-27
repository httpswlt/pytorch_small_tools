import os
import logging
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class DatasetsFactory(object):

    IMAGENET = "IMAGENET"
    CIFAR10 = "CIFAR10"
    COCO2014 = "coco2014"
    COCO2017 = "coco2017"
    VOC2007 = "VOC2007"
    VOC2012 = "VOC2012"

    def __init__(self):
        pass

    @staticmethod
    def create_dataset(dataset_type, dataset_path, transforms_para, batch_size,
                       is_debug=False, distributed=False, workers=4):
        if dataset_type == DatasetsFactory.IMAGENET:
            if is_debug:
                train_dir = os.path.join(dataset_path, 'train_mini')
            else:
                train_dir = os.path.join(dataset_path, 'train')
            val_dir = os.path.join(dataset_path, 'val')
            train_dataset = datasets.ImageFolder(
                train_dir,
                transforms_para)
            if distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                num_workers=workers, pin_memory=True, sampler=train_sampler)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(val_dir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                ])),
                batch_size=32, shuffle=False,
                num_workers=workers, pin_memory=True)

            return train_loader, val_loader, train_sampler


class Partition(object):
    """ Dataset partitioning helper """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


if __name__ == '__main__':
    batch_size = 32
    data = '/home/lingc1/data/ImageNet2012'
    train_loader, val_loader, train_sampler = DatasetsFactory.\
        create_dataset(dataset_type=DatasetsFactory.IMAGENET,
                       batch_size=batch_size,
                       dataset_path=data,
                       transforms_para=None,
                       distributed=False,
                       workers=24,
                       is_debug=True)

