# coding:utf-8
import sys
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import init
import time
from torch.autograd import Variable
sys.path.insert(0, "../")
from models.ssd import SSD
from module.prior_box import PriorBox
from loss.multibox_loss import MultiBoxLoss
from data_tools.load_data_voc import LoadVocDataSets, AnnotationTransform
from data_tools.load_data_voc import PreProcess, detection_collate
from utils.tools import adjust_learning_rate

voc = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'image_size': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        # xavier(m.weight.data)
        # xavier(m.bias.data)
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0.1)

# def weights_init(m):
#
#
#     for key in m.state_dict():
#         if key.split('.')[-1] == 'weight':
#             if 'conv' in key:
#                 # init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
#                 init.xavier_uniform()
#             if 'bn' in key:
#                 m.state_dict()[key][...] = 1
#         elif key.split('.')[-1] == 'bias':
#             m.state_dict()[key][...] = 0


def main():
    lr = 5e-4
    gamma = 0.2
    num_classes = 21
    epoch = 300
    batch_size = 32
    # data_path = '/mnt/storage/project/data/VOCdevkit/VOC2007'
    data_path = '~/datasets/VOC/VOCdevkit/VOC2007'

    # define data.
    data_set = LoadVocDataSets(data_path, 'trainval', AnnotationTransform(), PreProcess())

    # generate default bbox
    priors = PriorBox(voc)
    prior_box = priors.forward().cuda()

    # define network.
    ssd = SSD(image_size=300, num_classes=num_classes).cuda()
    ssd.apply(weights_init)
    print(ssd)
    # load pretrain model
    # ssd.vgg.load_state_dict(torch.load("../premodel/vgg16_reducedfc.pth"))
    # ssd.load_state_dict(torch.load('./ssd_epoches_4524.pth'))

    # define loss function
    criterion = MultiBoxLoss(num_classes=num_classes, overlap_thresh=0.5, prior_for_matching=True,
                             bkg_label=0, neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False)
    # define optimizer
    optimizer = optim.SGD(ssd.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # set iteration numbers.
    epoch_size = len(data_set) // batch_size
    max_iter = epoch_size * epoch

    adjust = 0
    # start iteration
    for iteration in range(max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iter = iter(DataLoader(data_set, batch_size, shuffle=True,
                                         num_workers=6, collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            torch.save(ssd.state_dict(), 'ssd_epoches_' + repr(iteration) + '.pth')

        # auto adjust lr
        if (iteration / float(epoch_size)) % 50 == 0:
            lr_ = adjust_learning_rate(lr, optimizer, gamma, epoch, adjust, iteration, epoch_size)
            adjust += 1

        # count time
        load_t0 = time.time()

        # load data
        images, targets = next(batch_iter)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

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
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % optimizer.param_groups[0]['lr'])#lr)


if __name__ == '__main__':
    main()




