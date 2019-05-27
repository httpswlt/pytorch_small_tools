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
import apex
from lars import LARS
from Alex import AlexNet
from load_cifar10_data import LoadClassifyDataSets, collate_fn


class ApexDistributeModel(object):
    def __init__(self, model, criterion, optimizer, config, gpu=None):
        super(ApexDistributeModel, self).__init__()
        self.model = model
        self.config = config
        self.sync_bn = self.config['sync_bn']
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
        return self.model, self.criterion, self.optimizer

    def lars(self):
        self.optimizer = LARS(self.optimizer)

    def train(self, epoch, train_loader):
        """
        you must run it after the 'convert' function.
        :param epoch:
        :param train_loader:
        :return:
        """
        self.model.train()
        print("Epoch is {}".format(epoch))
        train_iter = iter(train_loader)
        inputs, target = next(train_iter)
        step = 0
        start_time = time.time()
        while inputs is not None:
            step += 1
            inputs = inputs.cuda(self.gpu, non_blocking=True)
            target = target.cuda(self.gpu, non_blocking=True)
            output = self.model(inputs)
            loss = self.criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            self.optimizer.zero_grad()
            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()
            inputs, target = next(train_iter, (None, None))
            if step % 10 == 0:
                end_time = time.time()
                print("Step is {}, cost time: {}, loss: {}, acc1: {}, acc5:{}".
                      format(step, (end_time - start_time), loss.item(), acc1.item(), acc5.item()))
                start_time = time.time()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def run(gpu, config):
    cudnn.benchmark = True
    if config['distribute']:
        rank = config['rank'] * config['last_node_gpus'] + gpu
        print("world_size: {}, rank: {}".format(config['world_size'], rank))
        dist.init_process_group(backend=config['backend'], init_method=config['ip'],
                                world_size=config['world_size'], rank=rank)
    assert cudnn.enabled, "Amp requires cudnn backend to be enabled."
    torch.cuda.set_device(gpu)

    # create model
    model = AlexNet(10)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer strategy
    optimizer = torch.optim.SGD(model.parameters(), config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'])

    # convert pytorch to apex model.
    apexparallel = ApexDistributeModel(model, criterion, optimizer, config, gpu)
    apexparallel.convert()
    apexparallel.lars()

    # load data
    data_path = '/home/lintaowx/datasets/cifar10/train'
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
        apexparallel.train(epo, train_loader)


def main():
    config = {
        'ip': 'tcp://172.16.123.114:10000',
        'distribute': True,
        'world_size': 5,
        'rank': 1,
        'last_node_gpus': 1,
        'backend': 'nccl',
        'sync_bn': True,


        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.001,
        'batch_size': 1024,
        'num_workers': 4,
        'epoch': 100,

    }

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(run, nprocs=ngpus_per_node, args=(config, ))


if __name__ == '__main__':
    main()










