import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from model.darknet_224 import Darknet53, Darknet_cfg_53
from datasets.datasets_factory import DatasetsFactory
from utils import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate_epoch_poly


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', '--data', default='~/datasets/ImageNet2012', type=str, metavar='P',
# parser.add_argument('-data', '--data', default='/ifs/training_data/ImageNet2012', type=str, metavar='P',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='debug training')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main():
    args = parser.parse_args()
    args.gpu = 0
    model_cfg = os.path.join('model', 'darknet53.cfg')
    darknet_53_cfg = Darknet_cfg_53(cfgfile=model_cfg, num_classes=1000)

    model = darknet_53_cfg.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train_loader, val_loader, train_sampler = DatasetsFactory.create_dataset(dataset_type=DatasetsFactory.IMAGENET,
                                                                             batch_size=args.batch_size,
                                                                             dataset_path=args.data,
                                                                             transforms_para=Darknet53.transforms_para,
                                                                             distributed=False,
                                                                             workers=args.workers,
                                                                             is_debug=args.debug)
    train_index = -1

    dataset_conver_batches = len(train_loader.dataset) / args.batch_size
    max_batches = args.epochs * dataset_conver_batches

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args, max_batches=max_batches, train_index=train_index)


def train(train_loader, model, criterion, optimizer, epoch, args, max_batches, train_index=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    train_log_file = 'train_log'

    end = time.time()
    epoch_batch = epoch * (len(train_loader.dataset) / args.batch_size)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if train_index != -1 and i < train_index:
            continue
        adjust_learning_rate_epoch_poly(optimizer, args, burn_in=Darknet53.burn_in, power=Darknet53.power,
                                        batch_num=i + epoch_batch, max_batches=max_batches)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

        if i % 1000 == 0:
            save_checkpoint({
                'epoch': epoch,
                'index': i,
                'arch': "Darknet_53",
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, filename='darknet_53_init_55_pytorch_train_tmp.pth.tar')

    with open(train_log_file, 'a+') as log_file:
        log_file.write('Epoch:{0}, Loss{loss.avg:.4f}, Top1:{top1.avg:.3f},Top5:{top5.avg:.3f}\n'.format(
            epoch, loss=losses, top1=top1, top5=top5))


if __name__ == '__main__':
    main()