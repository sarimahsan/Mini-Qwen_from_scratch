import torch
import torch.nn as nn


class RMSProp:
    def __init__(self, params, lr=1e-2, beta=0.9, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.eps = eps

        self.v = [
            torch.zeros_like(p)
            for p in self.params
        ]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.v):

            if p.grad is None:
                continue

            g = p.grad

            v.mul_(self.beta).addcmul_(
                g, g,
                value=1 - self.beta
            )

            p.addcdiv_(
                g,
                v.sqrt().add(self.eps),
                value=-self.lr
            )