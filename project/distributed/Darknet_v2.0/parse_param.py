# coding:utf-8
import argparse


def parser_param():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # parser.add_argument('-data', '--data', default='/ifs/training_data/ImageNet2012', type=str, metavar='P',
    parser.add_argument('--data', '--data', default='/home/lingc1/data/ImageNet2012', type=str, metavar='P',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    # parser.add_argument('--lr', '--learning-rate', default=0.025, type=float,
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='debug training')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')

    # distribute parameters
    parser.add_argument('--backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--url', default='tcp://127.0.0.1:12345', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--last-node-gpus', default=0, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--sync_bn', action='store_true', default=False,
                        help='synchronization the batch normal on different GPUs.')
    parser.add_argument('--distribute', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    return args
