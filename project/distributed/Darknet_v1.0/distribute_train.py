import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import apex
import os
from lars import LARS
from model.darknet_224 import Darknet_cfg_53, Darknet53
from parse_param import parser_param
from datasets.datasets_factory import DatasetsFactory
from utils import AverageMeter, adjust_learning_rate_epoch_poly, save_checkpoint, accuracy


class ApexDistributeModel(object):
    def __init__(self, model, criterion, optimizer, args, gpu=None):
        super(ApexDistributeModel, self).__init__()
        self.model = model
        self.args = args
        self.sync_bn = self.args.sync_bn
        self.gpu = gpu
        if self.gpu is not None:
            assert isinstance(self.gpu, int), "GPU should is a int type."
        self.criterion = criterion
        self.optimizer = optimizer
        self.opt_level = None

    def convert(self, opt_level='O0'):
        self.opt_level = opt_level
        if self.sync_bn:
            # synchronization batch normal
            self.model = apex.parallel.convert_syncbn_model(self.model)
        # assign specific gpu
        self.model = self.model.cuda(self.gpu)
        self.criterion = self.criterion.cuda(self.gpu)
        # init model and optimizer by apex
        self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer,
                                                         opt_level=self.opt_level)
        # apex parallel
        self.model = apex.parallel.DistributedDataParallel(self.model, delay_allreduce=True)
        # self.model = apex.parallel.DistributedDataParallel(self.model)

        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu])
        return self.model, self.criterion, self.optimizer

    def lars(self):
        print("Enable LARS Optimizer Algorithm")
        self.optimizer = LARS(self.optimizer)

    def train(self, epoch, train_loader, max_batches, train_index):
        """
        you must run it after the 'convert' function.
        :param epoch:
        :param train_loader:
        :return:
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.train()

        train_log_file = 'train_log'

        end = time.time()
        epoch_batch = epoch * (len(train_loader.dataset) / self.args.batch_size)
        for i, (inputs, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if train_index != -1 and i < train_index:
                continue
            adjust_learning_rate_epoch_poly(self.optimizer, self.args, burn_in=Darknet53.burn_in, power=Darknet53.power,
                                            batch_num=i + epoch_batch, max_batches=max_batches)
            inputs = inputs.cuda(self.gpu, non_blocking=True)
            target = target.cuda(self.gpu, non_blocking=True)

            # compute output
            output = self.model(inputs)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            # loss.backward()
            time1 = time.time()
            with apex.amp.scale_loss(loss, self.optimizer) as scale_loss:
                scale_loss.backward()
            time2 = time.time()
            self.optimizer.step()
            time3 = time.time()
            print("step cost time: {}, backward cost time: {}".format(time3-time2, time2-time1))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))

            if i % 1000 == 0 and i:
                save_checkpoint({
                    'epoch': epoch,
                    'index': i,
                    'arch': "Darknet_53",
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, False, filename='darknet_53_init_55_pytorch_train_tmp.pth.tar')

        with open(train_log_file, 'a+') as log_file:
            log_file.write('Epoch:{0}, Loss{loss.avg:.4f}, Top1:{top1.avg:.3f},Top5:{top5.avg:.3f}\n'.format(
                epoch, loss=losses, top1=top1, top5=top5))


def process(gpu, args):
    cudnn.benchmark = True
    if args.distribute:
        rank = args.rank * args.last_node_gpus + gpu
        print("world_size: {}, rank: {}".format(args.world_size, rank))
        dist.init_process_group(backend=args.backend, init_method=args.url,
                                world_size=args.world_size, rank=rank)
    assert cudnn.enabled, "Amp requires cudnn backend to be enabled."
    torch.cuda.set_device(gpu)

    # create model
    model_cfg = os.path.join('model', 'darknet53.cfg')
    model = Darknet_cfg_53(cfgfile=model_cfg, num_classes=1000)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer strategy
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # convert pytorch to apex model.
    apexparallel = ApexDistributeModel(model, criterion, optimizer, args, gpu)
    apexparallel.convert()
    apexparallel.lars()

    print("=> creating model '{}'".format(model.model_name()))
    # load data
    train_loader, val_loader, train_sampler = DatasetsFactory.create_dataset(dataset_type=DatasetsFactory.IMAGENET,
                                                                             batch_size=args.batch_size,
                                                                             dataset_path=args.data,
                                                                             transforms_para=Darknet53.transforms_para,
                                                                             distributed=args.distribute,
                                                                             workers=args.workers,
                                                                             is_debug=args.debug)

    train_index = -1

    dataset_conver_batches = len(train_loader.dataset) / (args.batch_size * args.world_size)
    max_batches = args.epochs * dataset_conver_batches
    # start training
    for epo in range(args.start_epoch, args.epochs):
        if args.distribute:
            train_sampler.set_epoch(epo)

        # train for per epoch
        apexparallel.train(epo, train_loader, max_batches, train_index)


def main():
    args = parser_param()
    print(args)

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(process, nprocs=ngpus_per_node, args=(args, ))


if __name__ == '__main__':
    os.environ.__setitem__("CUDA_VISIBLE_DEVICES", "0")
    main()










