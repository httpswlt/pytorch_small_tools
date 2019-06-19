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

        bin_count = torch.zeros((self.bins,))
        for i in range(self.bins):
            bin_count[i] = (bin_idx == i).sum().item()

        n = x.size(0) * x.size(1)

        if self._last_bin_count is not None:
            bin_count = self.alpha * self._last_bin_count + (1 - self.alpha) * bin_count
        self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = n / gd
        return self._custom_loss(x, target, beta[bin_idx])

    def _g2bin(self, g):
        return torch.floor(g * (self.bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError


class GHMCLoss(GHMLoss):
    def __init__(self, bins, alpha):
        super(GHMLoss, self).__init__()
        self.bins = bins
        self.alpha = alpha
        self._last_bin_count = None

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return x.sigmoid().detach() - target


def main():
    pred = torch.FloatTensor([[1., 0., 0.5, 0.]])
    target = torch.FloatTensor([[1., 0., 0., 1.]])
    mask = torch.FloatTensor([[1., 1., 1., 1.]])
    ghmc = GHMCLoss(10, 0.75)
    print ghmc(pred, target)

if __name__ == '__main__':
    main()