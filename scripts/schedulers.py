import math
from torch.optim.lr_scheduler import _LRScheduler


class InvSqrtDecay(_LRScheduler):
    def __init__(
        self,
        optimizer,
        lambda_val,
        dim,
        warmup_steps,
        last_epoch=-1,
        verbose=False
    ):
        self.lambda_val = lambda_val
        self.dim = dim
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        lrs = []
        n = self.last_epoch + 1
        for base_lr in self.base_lrs:
            mult_coeff = self.lambda_val / math.sqrt(self.dim)
            learning_rate = mult_coeff * min(1 / math.sqrt(n), n / self.warmup_steps ** 1.5)
            lrs.append(learning_rate)
        return lrs
    

class Cosine(_LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_steps,
            max_steps,
            min_lr,
            max_lr,
            last_epoch=-1,
            verbose=False
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup_steps:
                lr = self.min_lr + (self.max_lr - self.min_lr) * \
                    (self.last_epoch / self.warmup_steps)
            else:
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                    (1 + math.cos(((self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)) * math.pi))
            lrs.append(lr)
        return lrs
