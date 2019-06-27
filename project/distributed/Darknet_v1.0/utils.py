# coding:utf-8
import torch
import shutil
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate_epoch_poly(optimizer, args, burn_in, power, batch_num, max_batches):
    if batch_num < burn_in:
        lr = args.lr * pow(float(batch_num) / burn_in, power)
    else:
        lr = args.lr * pow(1 - float(batch_num) / max_batches, power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return str(batch_num) + "   LR:  " + str(lr) + "\n"


def read_imagenet_labels_list(label_list_path):
    with open(label_list_path, 'r') as fl:
        cls_index = 0
        darknet_cls_index = {}
        for c_line in fl:
            if c_line != "":
                darknet_cls_index[c_line.strip('\n')] = cls_index
                cls_index += 1
    return darknet_cls_index


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,), darknet_class_list=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        if darknet_class_list is not None:
            pred_np = pred.cpu().numpy()
            pred_np_reshape = pred_np.reshape(pred_np.size)
            pred_np_torch = np.array([darknet_class_list[ci] for ci in pred_np_reshape])
            pred_np_torch = pred_np_torch.reshape(pred.shape)
            pred_np_torch = torch.from_numpy(pred_np_torch)
            target_cpu = target.cpu()
            correct = pred_np_torch.eq(target_cpu.view(1, -1).expand_as(pred_np_torch))
        else:
            correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res