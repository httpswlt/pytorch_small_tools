# coding: utf-8
import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
        reference:
            torch.optim.SGD and https://arxiv.org/abs/1708.03888
        Example:
            # >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
            # >>> optimizer.zero_grad()
            # >>> loss_fn(model(input), target).backward()
            # >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9, dampening=0, weight_decay=.0005,
                 eta=0.001, power=0, max_epoch=200, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening,
                        eta=eta, power=power, max_epoch=max_epoch, nesterov=nesterov)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            power = group['power']
            max_epoch = group['max_epoch']
            nesterov = group['nesterov']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # polynomial decay schedule
                decay = (1 - float(self.epoch) / max_epoch) ** power
                global_lr = lr * decay

                # compute local_lr for current layer
                local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)

                # update lr by LARS schedule
                update_lr = local_lr * global_lr

                # update gradient
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-update_lr, d_p)
        self.epoch += 1
        return loss
