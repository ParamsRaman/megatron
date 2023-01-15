import math
import torch
from torch.optim.optimizer import Optimizer


class MyLAMB2(Optimizer):

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
        super(MyLAMB2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MyLAMB2, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_d_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                diff = grad - exp_avg
                exp_d_sq = state['exp_d_sq']
                exp_d_sq.mul_(beta2).addcmul_(1 - beta2, diff, diff)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                denom_d = (exp_d_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                if state['step'] > 1:
                    r = exp_avg / bias_correction1 / denom
                    coeff = p.data.norm(2).clamp(min=1e-5 * p.numel()) / r.norm(2)
                r_d = diff / bias_correction1 / denom_d
                coeff_d = p.data.norm(2).clamp(min=1e-5 * p.numel()) / r_d.norm(2)

                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                if state['step'] > 1:
                    p.data.add_(r, alpha=-group['lr'] * coeff)
                p.data.add_(r_d, alpha=-group['lr'] * coeff_d)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        return loss
