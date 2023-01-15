import math
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist

from megatron import mpu


class LAMBTP(Optimizer):
    # solution 0: only use local weight and update norm
    # solution 1: S1 in our slides, communicate weight and update norm
    # solution 2: S2 in our slides, communicate weight norm, use local update norm
    # solution 3: S3 in our slides, communicate weight and stale update norm
    # no clamp is applied

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=1e-2, solution=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, solution=solution)
        super(LAMBTP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LAMBTP, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            solution = group['solution']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if not 'step' in state:
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
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                r = exp_avg / bias_correction1 / denom
                delta = r + group['weight_decay'] * p.data

                # local information only
                if solution == 0:
                    coeff = p.data.norm(2) / delta.norm(2)

                # s2 and s3
                if solution in [2, 3]:
                    if state['step'] == 1: 
                        coeff = p.data.norm(2).clamp(min=1e-5 * p.numel()) / delta.norm(2)
                    else:
                        # all reduce weight norm, stale delta norm (solution 3)
                        if solution == 3:
                            coeff = state['w2'].sqrt() / state['delta2'].sqrt()

                        # all reduce weight norm, local delta norm (solution 2)
                        if solution == 2:
                            state['w2'] /= mpu.get_tensor_model_parallel_world_size()
                            coeff = state['w2'].sqrt() / delta.norm(2)

                # all reduce weight norm, all reduce delta norm (solution 1)
                if solution == 1:
                    tp_group = mpu.get_tensor_model_parallel_group() 
                    tp_size = mpu.get_tensor_model_parallel_world_size()
                    norm1 = p.data.pow(2).sum()
                    norm2 = delta.pow(2).sum()
                    dist.all_reduce(norm1, group=tp_group)
                    dist.all_reduce(norm2, group=tp_group)
                    coeff = norm1.sqrt() / norm2.sqrt()

                p.data.add_(delta, alpha=-group['lr'] * coeff)

                if solution in [2, 3]:
                    state['delta2'] = delta.pow(2).sum()
                    state['w2'] = p.data.pow(2).sum()

        if solution in [2, 3]:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    tp_group = mpu.get_tensor_model_parallel_group()
                    dist.all_reduce(state['delta2'], group=tp_group)
                    dist.all_reduce(state['w2'], group=tp_group)

        return loss
