# coding:utf-8
from Alex import AlexNet
from load_cifar10_data import LoadClassifyDataSets, collate_fn
from lars import LARS
import torch
import apex
from torch import nn
from torch.utils.data import DataLoader, distributed
import time
torch.backends.cudnn.benchmark = True


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.long().cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def main(is_distributed, rank, ip, sync_bn):
    world_size = 1
    if is_distributed:
        world_size = 2
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=ip,
                                             world_size=world_size,
                                             rank=rank)
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    print("Connect")
    # set hyper parameters
    batch_size = 128
    lr = 0.01  # base on batch size 256
    momentum = 0.9
    weight_decay = 0.0001
    epoch = 100

    # recompute lr
    lr = lr * world_size

    # create model
    model = AlexNet(10)

    # synchronization batch normal
    if sync_bn:
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer strategy
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    model, optimizer = apex.amp.initialize(model, optimizer,
                                           opt_level='O0')
    optimizer = LARS(optimizer)

    if is_distributed:
        # for distribute training
        model = nn.parallel.DistributedDataParallel(model)
        # model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    # load train data
    data_path = '~/datasets/cifar10/train'
    train_set = LoadClassifyDataSets(data_path, 227)
    train_sampler = None
    if is_distributed:
        train_sampler = distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size, shuffle=(train_sampler is None),
                              num_workers=4, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    for epoch in range(100):
        # for distribute
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        # train_iter = iter(train_loader)
        # inputs, target = next(train_iter)
        prefetcher = DataPrefetcher(train_loader)
        inputs, target = prefetcher.next()

        step = 0
        print("Epoch is {}".format(epoch))
        while inputs is not None:
            step += 1
            print("Step is {}".format(step))

            time_model_1 = time.time()
            output = model(inputs)
            time_model_2 = time.time()
            print("model time: {}".format(time_model_2 - time_model_1))
            time_loss_1 = time.time()
            loss = criterion(output, target.cuda(async=True))
            time_loss_2 = time.time()
            print("loss time: {}".format(time_loss_2 - time_loss_1))
            optimizer.zero_grad()
            time_back_1 = time.time()
            # loss.backward()
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            time_back_2 = time.time()
            print("back time: {}".format(time_back_2 - time_back_1))
            optimizer.step()
            # if step % 10 == 0:
            #     print("loss is : {}", loss.item())
            # inputs, target = next(train_iter, (None, None))
            inputs, target = prefetcher.next()


if __name__ == '__main__':
    is_distributed = True
    sync_bn = True
    rank = 0
    ip = 'tcp://172.16.123.114:10000'
    # ip = 'tcp://172.16.117.110:10000'
    main(is_distributed, rank, ip, sync_bn)




