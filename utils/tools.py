# coding:utf-8


def adjust_learning_rate(lr, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr_ = 1e-6 + (lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr_ = lr * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr_
