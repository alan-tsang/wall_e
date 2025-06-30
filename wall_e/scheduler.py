import math
from .common.registry import registry


@registry.register_lr_scheduler("LinearWarmupStepLRScheduler")
class LinearWarmupStepLRScheduler:
    def __init__(
            self,
            optimizer,
            max_epoch,
            iters_per_epoch,
            min_lr,
            max_lr,
            decay_rate = 0.9,
            warmup_rate = 0.05,
            warmup_start_lr = 2e-5,
            **kwargs
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.decay_rate = decay_rate
        self.warmup_rate = warmup_rate
        self.warmup_start_lr = warmup_start_lr

        # Calculated parameters
        self.max_steps = max_epoch * iters_per_epoch
        self.warmup_steps = int(warmup_rate * self.max_steps)

        # Training state
        self.current_epoch = 0
        self.current_step = 0

        # Logger
        self.logger = registry.get("runner.logger")

    def step(self, cur_epoch, cur_step):
        # Update training state
        self.current_epoch = cur_epoch
        self.current_step = cur_step

        if cur_epoch == 0:
            lr = warmup_lr_schedule(
                step = cur_step,
                optimizer = self.optimizer,
                max_step = self.warmup_steps,
                warmup_start_lr = self.warmup_start_lr,
                max_lr = self.max_lr,
            )
        else:
            lr = step_lr_schedule(
                epoch = cur_epoch,
                optimizer = self.optimizer,
                max_lr = self.max_lr,
                min_lr = self.min_lr,
                decay_rate = self.decay_rate,
            )
        self.logger.info(f"当前学习率: {lr:.6f} at epoch {cur_epoch}, step {cur_step}")

    def state_dict(self):
        return {
            'max_epoch': self.max_epoch,
            'iters_per_epoch': self.iters_per_epoch,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'decay_rate': self.decay_rate,
            'warmup_rate': self.warmup_rate,
            'warmup_start_lr': self.warmup_start_lr,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict):
        self.max_epoch = state_dict['max_epoch']
        self.iters_per_epoch = state_dict['iters_per_epoch']
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.decay_rate = state_dict.get('decay_rate', 0.9)  # Backward compatibility
        self.warmup_rate = state_dict['warmup_rate']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.current_epoch = state_dict['current_epoch']
        self.current_step = state_dict['current_step']

        # Recalculate derived parameters
        self.max_steps = self.max_epoch * self.iters_per_epoch
        self.warmup_steps = int(self.warmup_rate * self.max_steps)


@registry.register_lr_scheduler("LinearWarmupCosineLRScheduler")
class LinearWarmupCosineLRScheduler:
    def __init__(
            self,
            optimizer,
            max_epoch,
            iters_per_epoch,
            min_lr,
            max_lr,
            warmup_rate = 0.05,
            warmup_start_lr = 2e-5,
            **kwargs
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_rate = warmup_rate
        self.warmup_start_lr = warmup_start_lr

        # Calculated parameters
        self.max_steps = max_epoch * iters_per_epoch
        self.warmup_steps = int(warmup_rate * self.max_steps)

        # Training state
        self.current_step = 0

        # Logger
        self.logger = registry.get("runner.logger")

    def step(self, cur_step):
        # Update training state
        self.current_step = cur_step

        if cur_step < self.warmup_steps:
            lr = warmup_lr_schedule(
                step = cur_step,
                optimizer = self.optimizer,
                max_step = self.warmup_steps,
                warmup_start_lr = self.warmup_start_lr,
                max_lr = self.max_lr,
            )
        else:
            lr = cosine_lr_schedule(
                step = cur_step - self.warmup_steps,
                optimizer = self.optimizer,
                max_steps = self.max_steps - self.warmup_steps,
                max_lr = self.max_lr,
                min_lr = self.min_lr,
            )
        self.logger.info(f"当前学习率: {lr:.6f} at step {cur_step}")

    def state_dict(self):
        return {
            'max_epoch': self.max_epoch,
            'iters_per_epoch': self.iters_per_epoch,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'warmup_rate': self.warmup_rate,
            'warmup_start_lr': self.warmup_start_lr,
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict):
        self.max_epoch = state_dict['max_epoch']
        self.iters_per_epoch = state_dict['iters_per_epoch']
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.warmup_rate = state_dict['warmup_rate']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.current_step = state_dict['current_step']

        # Recalculate derived parameters
        self.max_steps = self.max_epoch * self.iters_per_epoch
        self.warmup_steps = int(self.warmup_rate * self.max_steps)


def cosine_lr_schedule(optimizer, step, max_steps, max_lr, min_lr):
    """Decay the learning rate"""
    lr = (max_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * step / max_steps)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def warmup_lr_schedule(optimizer, step, max_step, warmup_start_lr, max_lr):
    """Warmup the learning rate"""
    lr = warmup_start_lr + (max_lr - warmup_start_lr) * step / max(max_step, 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def step_lr_schedule(optimizer, epoch, max_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, max_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
