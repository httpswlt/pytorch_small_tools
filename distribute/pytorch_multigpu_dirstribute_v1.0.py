import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import distributed, DataLoader
from Alex import AlexNet
from load_cifar10_data import LoadClassifyDataSets, collate_fn


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config['lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def run(gpu, config):
    cudnn.benchmark = True
    if config['distribute']:
        rank = config['rank'] * config['last_node_gpus'] + gpu
        print("world_size: {}, rank: {}".format(config['world_size'], rank))
        dist.init_process_group(backend=config['backend'], init_method=config['ip'],
                                world_size=config['world_size'], rank=rank)

    # create model
    model = AlexNet(10)

    if config['distribute']:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer strategy
    optimizer = torch.optim.SGD(model.parameters(), config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'])

    # load data
    data_path = '~/datasets/cifar10/train'
    train_set = LoadClassifyDataSets(data_path, 227)
    train_sampler = None
    if config['distribute']:
        train_sampler = distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, config['batch_size'], shuffle=(train_sampler is None),
                              num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler,
                              collate_fn=collate_fn)

    for epo in range(config['epoch']):
        if config['distribute']:
            train_sampler.set_epoch(epo)

        # train for per epoch
        train(train_loader, model, criterion, optimizer, epo, gpu)


def train(train_loader, model, criterion, optimizer, epoch, gpu):
    model.train()

    print("Epoch is {}".format(epoch))
    train_iter = iter(train_loader)
    inputs, target = next(train_iter)

    step = 0
    while inputs is not None:

        inputs = inputs.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)
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
        loss.backward()
        time_back_2 = time.time()
        print("back time: {}".format(time_back_2 - time_back_1))
        optimizer.step()
        # if step % 10 == 0:
        #     print("loss is : {}", loss.item())
        inputs, target = next(train_iter, (None, None))


def main():
    config = {
        'ip': 'tcp://172.16.123.114:10000',
        'distribute': True,
        'world_size': 5,
        'rank': 1,
        'last_node_gpus': 1,
        'backend': 'nccl',
        'gpu': None,    # [0, 1, ...]

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.001,
        'batch_size': 128,
        'num_workers': 4,
        'epoch': 100,

    }

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(run, nprocs=ngpus_per_node, args=(config, ))


if __name__ == '__main__':
    main()










