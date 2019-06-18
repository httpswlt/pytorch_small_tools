# coding:utf-8
"""
    inference: https://arxiv.org/pdf/1811.05181
"""
import torch
from torch import nn
import torch.nn.functional as F


class GHMLoss(nn.Module):
    def __init__(self):
        super(GHMLoss, self).__init__()
        self.bins = None
        self.alpha = None
        self._last_bin_count = None

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target))

        bin_idx = self._g2bin(g)
        print (123)

    def _g2bin(self, g):
        return torch.floor(g * (self.bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError


class GHMCLosss(GHMLoss):
    def __init__(self, bins, alpha):
        super(GHMLoss, self).__init__()
        self.bins = bins
        self.alpha = alpha

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


def main():
    pred = torch.FloatTensor([[1., 0., 0.5, 0.]])
    target = torch.FloatTensor([[1., 0., 0., 1.]])
    ghmc = GHMCLosss(10, 0.75)
    ghmc(pred, target)

if __name__ == '__main__':
    main()