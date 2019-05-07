# coding:utf-8
import sys
import torch
from torch.utils.data import DataLoader
from torch import optim
import time
sys.path.append("../")
from models.ssd import SSD
from module.prior_box import PriorBox
from loss.multibox_loss import MultiBoxLoss
from data_tools.load_data_voc import LoadVocDataSets, AnnotationTransform
from data_tools.load_data_voc import PreProcess, detection_collate
from torch.autograd import Variable

voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'image_size': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


def main():
    lr = 0.01
    num_classes = 21
    epoch = 30
    batch_size = 32
    # data_path = '/mnt/storage/project/data/VOCdevkit/VOC2007'
    data_path = '/home/lintaowx/datasets/VOC/VOCdevkit/VOC2007'

    # define data.
    data_set = LoadVocDataSets(data_path, 'trainval', AnnotationTransform(), PreProcess())

    # define default bbox
    priors = PriorBox(voc)
    prior_box = priors.forward().cuda()

    # define network.
    ssd = SSD(image_size=300, num_classes=num_classes).cuda()

    # define loss function
    criterion = MultiBoxLoss(num_classes=num_classes, overlap_thresh=0.5, prior_for_matching=True,
                             bkg_label=0, neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False)
    # define optimizer
    optimizer = optim.SGD(ssd.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # set iteration numbers.
    epoch_size = len(data_set) // batch_size
    max_iter = epoch_size * epoch

    # start iteration
    for iteration in range(max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iter = iter(DataLoader(data_set, batch_size, shuffle=True,
                                         num_workers=6, collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            torch.save(ssd.state_dict(), 'epoches_' + repr(iteration) + '.pth')

        # count time
        load_t0 = time.time()

        # load data
        images, targets = next(batch_iter)
        images = Variable(images.cuda())
        targets = [Variable(anno.cuda()) for anno in targets]

        # forward
        output = ssd.forward(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(output, prior_box, targets)
        # calculate loss
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        # calculate total error
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                loss_l.item(), loss_c.item()) +
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % lr)


if __name__ == '__main__':
    main()




