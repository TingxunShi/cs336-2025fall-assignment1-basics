import math
import torch

from collections.abc import Callable, Iterable
from typing import Optional


PI = 3.14159265359
EPSILON = 1e-6


def cosine_aneealing_lr_schedule(min_lr: float, max_lr: float, t: int,
                                 warmup_iters: int, cosine_annealing_iters: int) -> float:
    if t < warmup_iters:
        lr = t / warmup_iters * max_lr
    elif t > cosine_annealing_iters:
        lr = min_lr
    else:
        a = (1 + math.cos(PI * (t - warmup_iters) / (cosine_annealing_iters - warmup_iters)))
        b = (max_lr - min_lr)
        lr = min_lr + 0.5 * a * b
    return lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += torch.sum(p.grad.data ** 2)
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + EPSILON)
    if clip_coef > 1:
        return
    for p in parameters:
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {
            'alpha': lr,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group['alpha']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]

                # lazy init
                if not state:
                    state['m'] = torch.zeros_like(param.data)
                    state['v'] = torch.zeros_like(param.data)
                    state['t'] = 0

                m, v, t = state['m'], state['v'], state['t']
                t += 1
                m.mul_(beta1).add_(param.grad.data, alpha=1.0 - beta1)
                v.mul_(beta2).add_(torch.pow(param.grad.data, 2.0), alpha=1.0 - beta2)
                adjusted_alpha = alpha * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                param.data -= adjusted_alpha * m / (torch.sqrt(v) + eps)
                param.data -= alpha * weight_decay * param.data
                state['t'] = t

        return loss
