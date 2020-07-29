# coding: utf-8
import sys
sys.path.insert(0, '..')
from data_tools.load_data_voc import LoadVocDataSets, AnnotationTransform
from data_tools.load_data_voc import PreProcess, detection_collate
from models.yolov3 import YOLOV3
from loss.yolo_v3_loss import YOlOLoss
from parameter import conf_yolov3
from torch import optim
from torch.utils.data import DataLoader
from utils.tools import adjust_learning_rate
import time


def main():
    # load parameter
    lr = conf_yolov3['lr']
    momentum = conf_yolov3['momentum']
    weight_decay = conf_yolov3['weight_decay']
    batch_size = conf_yolov3['batch_size']
    epoch = conf_yolov3['epoch']
    gamma = conf_yolov3['gamma']
    # load data
    # data_path = '/mnt/storage/project/data/VOCdevkit/VOC2007'
    data_path = '~/datasets/VOC/VOCdevkit/VOC2007'
    data_set = LoadVocDataSets(data_path, 'trainval', AnnotationTransform(), PreProcess(resize=(416, 416)))

    # define network.
    yolov3 = YOLOV3(conf_yolov3).cuda()
    print(yolov3)

    # define loss function.
    yolo_losses = [YOlOLoss(conf_yolov3, i).cuda() for i in range(3)]

    # define optimize function
    optimizer = optim.SGD(yolov3.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # set iteration numbers.
    epoch_size = len(data_set) // batch_size
    max_iter = epoch_size * epoch

    adjust = 0
    # start iteration
    for iteration in range(max_iter):
        if iteration % epoch_size == 0:
            # recreate batch iterator
            batch_iter = iter(DataLoader(data_set, batch_size, shuffle=True,
                                         num_workers=6, collate_fn=detection_collate))

        # auto adjust lr
        if (iteration / float(epoch_size)) % (epoch / 3) == 0:
            lr_ = adjust_learning_rate(lr, optimizer, gamma, epoch, adjust, iteration, epoch_size)

        # count time
        load_t0 = time.time()

        # generate data
        images, targets = next(batch_iter)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        outputs = yolov3(images)

        # backward
        optimizer.zero_grad()
        loss = []
        for i, output in enumerate(outputs):
            loss.append(yolo_losses[i](output, targets))
        total_loss = sum(loss)
        total_loss.backward()
        optimizer.step()
        load_t1 = time.time()

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' ||Loss: %.4f||' % total_loss +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % optimizer.param_groups[0]['lr'])#lr)


if __name__ == '__main__':
    main()

