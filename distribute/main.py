# coding:utf-8
from Alex import AlexNet
from load_cifar10_data import LoadClassifyDataSets, collate_fn
import torch
import apex
from torch import nn
from torch.utils.data import DataLoader, distributed
import tqdm
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


def adjust_learning_rate(learn_rate, optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = learn_rate * (0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(is_distributed, sync_bn, rank):
    world_size = 1
    if is_distributed:
        world_size = 2
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://172.16.117.110:1234',
                                             world_size=world_size,
                                             rank=rank)
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # set hyper parameters
    batch_size = 30
    lr = 0.01  # base on batch size 256
    momentum = 0.9
    weight_decay = 0.0001
    epoch = 100

    # recompute lr
    lr = lr * world_size

    # create model
    model = AlexNet(10)
    # leverage apex to realize batch_normal synchronization in different GPU
    # if sync_bn:
    #     model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer strategy
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # initialize Amp
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level='O0')
    if is_distributed:
        # for distribute training
        # model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        model = nn.parallel.DistributedDataParallel(model)
    # load train data
    data_path = '~/datasets/cifar10/train'
    train_set = LoadClassifyDataSets(data_path, 227)
    train_sampler = None
    if is_distributed:
        train_sampler = distributed.DistributedSampler(train_set, world_size, rank=rank)
        # train_sampler = distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size, shuffle=(train_sampler is None),
                              num_workers=4, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)


    for epoch in range(100):
        # for distribute
        if is_distributed:
            train_sampler.set_epoch(epoch)


        model.train()
        train_iter = iter(train_loader)
        inputs, target = next(train_iter)

        step = 0
        print("Epoch is {}".format(epoch))
        while inputs is not None:
            step += 1
            print("test0")
            temp = inputs.cuda()
            print("test01")
            output = model(temp)
            print("test1")
            loss = criterion(output, target.cuda(async=True))
            print("test2")
            optimizer.zero_grad()
            print("test3")

            loss.backward()
            print("test4")
            # with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            print("test5")
            if step % 10 == 0:
                print("loss is : ", loss.item())
            inputs, target = next(train_iter, (None, None))


if __name__ == '__main__':
    is_distributed = True
    sync_bn = True
    rank = 0
    main(is_distributed, sync_bn, rank)
