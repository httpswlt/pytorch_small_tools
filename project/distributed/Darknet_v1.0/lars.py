# coding: utf-8
import torch
from torch.optim.optimizer import Optimizer, required

'''
LAYER-WISE ADAPTIVE RATE SCALING(LARS)
Layer             |  conv1.b conv1.w conv2.b conv2.w conv3.b conv3.w conv4.b conv4.w
||w||             |  1.86    0.098   5.546   0.16    9.40    0.196   8.15    0.196
||∇L(w)||         |  0.22    0.017   0.165   0.002   0.135   0.0015  0.109   0.0013
||w|| / ||∇L(w)|| |  8.48    5.76    33.6    83.5    69.9    127     74.6    148
Layer             |  conv5.b conv5.w fc6.b   fc6.w   fc7.b   fc7.w   fc8.b   fc8.w
||w||             |  6.65    0.16    30.7    6.4     20.5    6.4     20.2    0.316
||∇L(w)||         |  0.09    0.0002  0.26    0.005   0.30    0.013   0.22    0.016
||w|| / ||∇L(w)|| |  73.6    69      117     1345    68      489     93      19
'''


class LARS(object):
    r"""Implements layer-wise adaptive rate scaling for SGD.
            reference:
            https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
            And https://arxiv.org/abs/1708.03888
            it must behind the optimizer, so you can use different optimizer before invoke it.

        Example:
            # >>> optimizer = torch.optim.SGD()|torch.optim.Adam()...
            # >>> optimizer = LARS(optimizer)
            # >>> optimizer.zero_grad()
            # >>> loss_fn(model(input), target).backward()
            # >>> optimizer.step()

        Args:
            optimizer: Pytorch optimizer
            trust_coefficient: Trust coefficient for calculating the lr
            clip: Decides between clipping or scaling mode of LARC. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
            eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """
    def __init__(self, optimizer, trust_coefficient=0.02, clip=False, eps=1e-8):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * param_norm / (
                                    grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

