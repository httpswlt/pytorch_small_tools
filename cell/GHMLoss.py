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
        self.edges = None
        self.acc_sum = None

    def forward(self, x, target):
        # calculate gradient by prediction.
        g = torch.abs(self._custom_loss_grad(x, target))

        # how many samples.
        n = x.size(0) * x.size(1)

        # the weights of x
        weights = torch.zeros_like(x)
        nonempty_bins = 0

        for i in range(self.bins):
            indx = self._gnormal(g, i)
            bin_count = indx.sum().item()
            if bin_count < 1:
                continue

            if self.alpha > 0:
                self.acc_sum[i] = self.alpha * self.acc_sum[i] + (1 - self.alpha) * bin_count
                weights[indx] = n / self.acc_sum[i]
            else:
                weights[indx] = n / bin_count
            nonempty_bins += 1

        if nonempty_bins > 0:
            weights = weights / nonempty_bins

        weights = torch.clamp(weights, min=0.0001)

        return self._custom_loss(x, target, weights)

    def _gnormal(self, g, i):
        return (g >= self.edges[i]) & (g < self.edges[i + 1])

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
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if self.alpha > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return x.sigmoid().detach() - target


def main():
    pred = torch.FloatTensor([[1., 0., 0.5, 0.]])
    target = torch.FloatTensor([[1., 0., 0., 1.]])
    mask = torch.FloatTensor([[1., 1., 1., 1.]])
    ghmc = GHMCLoss(10, 0.75)
    print(ghmc(pred, target))


if __name__ == '__main__':
    main()