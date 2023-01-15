import math
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import numpy as np


class MyLAMB3(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=1e-2, alpha=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, alpha=alpha)
        super(MyLAMB3, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MyLAMB3, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        max_numel = 0
        for group in self.param_groups:
            for p in group['params']:
                max_numel = max(max_numel, p.data.numel())

        idx = 0
        first_layer_idx = np.arange(0,7).tolist() + np.arange(22,30).tolist()
        last_layer_idx = np.arange(19,22).tolist() + np.arange(54,62).tolist()
        large1_idx = [0]
        large2_idx = [1, 5, 6, 9, 10, 13, 14, 17, 18]
        for group in self.param_groups:
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                r = exp_avg / denom + wd * p.data
                coeff = p.data.norm(2).clamp(min=1e-5 * p.numel()) / r.norm(2)
                #if idx in large1_idx:
                #    p.data.add_(r, alpha=-group['lr'] * coeff * group['alpha'])
                #else:
                #    p.data.add_(r, alpha=-group['lr'] * coeff)

                #coeff2 = 1.0 + p.numel() / max_numel * 1.0
                coeff2 = 1.0 + p.numel() / max_numel * group['alpha']
                #coeff2 = 2.0 if idx == 0 else 1.0
                #if dist.get_rank() == 0: print(state['step'], idx, coeff2, coeff2 * group['lr'], coeff)
                p.data.add_(r, alpha=-group['lr'] * coeff * coeff2)

                idx += 1

        return loss
