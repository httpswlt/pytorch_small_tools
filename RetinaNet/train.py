# coding:utf-8
from data_tools.load_data_voc import LoadVocDataSets, AnnotationTransform
from data_tools.load_data_voc import PreProcess, detection_collate
from model.retinanet import RetinaNet
from loss.focal_loss import FocalLoss
from torch import optim
from torch.utils.data import DataLoader


def main():
    lr = 5e-4
    gamma = 0.2
    num_classes = 21
    epoch = 300
    batch_size = 1
    # data_path = '/mnt/storage/project/data/VOCdevkit/VOC2007'
    data_path = '/home/lintaowx/datasets/VOC/VOCdevkit/VOC2007'

    # define data.
    data_set = LoadVocDataSets(data_path, 'trainval', AnnotationTransform(), PreProcess())

    # define model
    model = RetinaNet(num_classes)

    # define criterion
    criterion = FocalLoss(num_classes)

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # set iteration numbers.
    epoch_size = len(data_set) // batch_size
    max_iter = epoch_size * epoch

    train_loss = 0
    # start iteration
    for iteration in range(max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iter = iter(DataLoader(data_set, batch_size, shuffle=True,
                                         num_workers=6, collate_fn=detection_collate))
        images, loc_targets, cls_targets = next(batch_iter)
        optimizer.zero_grad()
        loc_preds, cls_preds = model(images)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f ' % (loss.data[0]))


if __name__ == '__main__':
    main()
