import torch


class AdamW:
    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # state
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1

        b1, b2 = self.betas

        for i, p in enumerate(self.params):

            if p.grad is None:
                continue

            g = p.grad

            # ----------------------
            # weight decay (decoupled)
            # ----------------------
            if self.weight_decay != 0:
                p.data.mul_(1 - self.lr * self.weight_decay)

            # ----------------------
            # first moment
            # ----------------------
            self.m[i].mul_(b1).add_(g, alpha=1 - b1)

            # ----------------------
            # second moment
            # ----------------------
            self.v[i].mul_(b2).addcmul_(g, g, value=1 - b2)

            # ----------------------
            # bias correction
            # ----------------------
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)

            # ----------------------
            # update
            # ----------------------
            p.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)