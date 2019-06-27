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
from lars import LARS, LARSOptimizer
from model.darknet_224 import Darknet_cfg_53, Darknet53
from parse_param import parser_param
from datasets.datasets_factory import DatasetsFactory
from utils import AverageMeter, adjust_learning_rate_epoch_poly, save_checkpoint, accuracy
import cProfile


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
        # self.criterion = self.criterion.cuda(self.gpu)
        # init model and optimizer by apex
        # self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer,
        #                                                  opt_level=self.opt_level)
        # apex parallel
        # self.model = apex.parallel.DistributedDataParallel(self.model, delay_allreduce=True)
        # self.model = apex.parallel.DistributedDataParallel(self.model)

        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], bucket_cap_mb=10)
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
            data_time.update(time.time() - end)
            print("data iteration cost time: {}ms".format(data_time.val * 1000))
            # print(target)
            gpu_1 = time.time()
            inputs = inputs.cuda(self.gpu, non_blocking=True)
            target = target.cuda(self.gpu, non_blocking=True)
            gpu_2 = time.time()
            print("convert datasets to gpu cost time: {}ms".format((gpu_2 - gpu_1) * 1000))

            inference_1 = time.time()
            output = self.model(inputs)
            inference_2 = time.time()
            print("inference time cost: {}ms".format((inference_2 - inference_1) * 1000))

            loss_1 = time.time()
            loss = self.criterion(output, target)
            loss_2 = time.time()
            print("loss cost time: {}ms".format((loss_2 - loss_1) * 1000))

            zero_1 = time.time()
            self.optimizer.zero_grad()
            zero_2 = time.time()
            print("zero cost time: {}ms".format((zero_2 - zero_1) * 1000))

            backward_1 = time.time()
            loss.backward()
            backward_2 = time.time()
            print("backward cost time: {}ms".format((backward_2 - backward_1) * 1000))

            step_1 = time.time()
            self.optimizer.step()
            step_2 = time.time()
            print("step cost time: {}ms".format((step_2 - step_1) * 1000))

            batch_time.update(time.time() - end)
            print("total cost time is: {}s".format(batch_time.val))
            # item_1 = time.time()
            # print("loss is {}".format(loss.item()))
            # item_2 = time.time()
            # print("loss item cost: {}s".format(item_2 - item_1))
            print("==================================")

            end = time.time()
            # if i == 10000:
            #     exit(0)


def process(gpu, args):
    cudnn.benchmark = True
    if args.distribute:
        rank = args.rank * args.last_node_gpus + gpu
        dist.init_process_group(backend=args.backend, init_method=args.url,
                                world_size=args.world_size, rank=rank)
        print("world_size: {}, rank: {}".format(dist.get_world_size(), dist.get_rank()))
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
    # optimizer = LARSOptimizer(model.parameters(), args.lr,
    #                           momentum=args.momentum,
    #                           weight_decay=args.weight_decay)

    # convert pytorch to apex model.
    apexparallel = ApexDistributeModel(model, criterion, optimizer, args, gpu)
    apexparallel.convert()
    # apexparallel.lars()

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
    # os.environ.__setitem__("CUDA_VISIBLE_DEVICES", "0")

    main()










