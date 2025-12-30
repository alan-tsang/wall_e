from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import contextlib

from .base_loop import BaseLoop
from ..util import move_data_to_device
from ...logging import print_log
from ...util.dl_util import get_batch_n
from ...common.registry import registry


class TrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
    """

    def __init__(
            self,
            runner: 'Runner',
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            valid_begin_epoch,
            valid_interval_epoch,
            valid_begin_iter,
            valid_interval_iter,
            test_begin_epoch,
            test_interval_epoch,
            test_begin_iter,
            test_interval_iter,
            shuffle = True,
    ):
        super().__init__(runner, dataloader, shuffle)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._iters_per_epoch = len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.valid_begin_epoch = valid_begin_epoch
        self.valid_interval_epoch = valid_interval_epoch
        self.valid_begin_iter = valid_begin_iter
        self.valid_interval_iter = valid_interval_iter
        self.test_begin_epoch = test_begin_epoch
        self.test_interval_epoch = test_interval_epoch
        self.test_begin_iter = test_begin_iter
        self.test_interval_iter = test_interval_iter

        self.cfg = self.runner.cfg
        # This attribute will be updated by `EarlyStopCallBack`
        # when it is enabled.

        self.scheduler = self.setup_scheduler()
        self.runner.scheduler = self.scheduler

        self.__post_init__()

    def __post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler() if self.is_16bit else None

    @property
    def max_epochs(self) -> int:
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self) -> int:
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._iter

    @property
    def is_16bit(self) -> bool:
        return OmegaConf.select(self.runner.cfg, "training.fp16", default = False)

    @property
    def accumulation_steps(self) -> int:
        return OmegaConf.select(
            self.runner.cfg, "training.gradient_accumulation",
            default = 1
        )

    @property
    def max_norm(self) -> Optional[float]:
        return OmegaConf.select(self.runner.cfg, "training.grad_clip", default = None)

    def prepare_run(self):
        if self.runner.state.resume_from:
            self.resume()
        # 在runner的setup_model，处理了
        # elif self.runner.state.load_from:
        #     self.load()

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.before_train()
        self.prepare_run()

        while self._epoch < self._max_epochs and not self.runner.state.stop_training:
            self.runner.state.current_epoch = self._epoch + 1
            self.run_epoch()

            if (self.runner.valid_loop is not None
                    and self._epoch >= self.valid_begin_epoch
                    and ((self._epoch - self.valid_interval_epoch) % self.valid_interval_epoch == 0
                         or self._epoch == self._max_epochs)):
                self.runner.logger.info(f"验证 at epoch: {self._epoch}...")
                self.runner.valid_loop.run()

            if (self.runner.test_loop is not None
                    and self._epoch >= self.test_begin_epoch
                    and ((self._epoch - self.test_interval_epoch) % self.test_interval_epoch == 0
                         or self._epoch == self._max_epochs)):
                self.runner.logger.info(f"测试 at epoch: {self._epoch}...")
                self.runner.test_loop.run()

        self.runner.after_train()
        return self.runner.model

    def run_epoch(self):
        """Iterate one epoch."""
        self.runner.before_running_epoch()
        self.runner.model.train()
        self._iter = 0

        for idx, data_batch in enumerate(self.dataloader):
            if self.runner.state.stop_training:
                break
            if isinstance(self.runner.train_data_loader.sampler, DistributedSampler):
                self.runner.train_data_loader.sampler.set_epoch(self.runner.state.current_epoch)
            data_batch = move_data_to_device(data_batch, self.runner.state.device)
            with self.maybe_autocast(self.is_16bit):
                self.runner.state.current_step = self._iter + 1
                self.run_iter(idx, data_batch)  # type: ignore
            self._iter += 1

            iter_now = self._iter + self._epoch * self._iters_per_epoch
            if (self.runner.valid_loop is not None
                    and iter_now >= self.valid_begin_iter
                    and ((iter_now - self.valid_begin_iter) % self.valid_interval_iter == 0
                         or iter_now == self._max_iters)):
                self.runner.logger.info(f"验证 at iter: {iter_now}...")
                self.runner.valid_loop.run()

            if (self.runner.test_loop is not None
                    and iter_now >= self.test_begin_iter
                    and ((iter_now - self.test_begin_iter) % self.test_interval_iter == 0
                         or iter_now == self._max_iters)):
                self.runner.logger.info(f"测试 at iter: {iter_now}...")
                self.runner.test_loop.run()

        self.runner.after_running_epoch()
        self._epoch += 1

    def run_iter(self, idx, data_batch: dict[str, Sequence]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.before_running_batch()

        if hasattr(self.runner.model, "module"):
            # For DataParallel or DistributedDataParallel
            model_output = self.runner.model.module.train_step(data_batch)
        else:
            model_output = self.runner.model.train_step(data_batch)
        assert "loss" in model_output, "模型输出必须返回包含loss的字典"

        skip_batch = torch.isnan(model_output["loss"]).any().item()

        # 分布式环境下同步跳过的决策
        if torch.distributed.is_initialized():
            skip_batch_tensor = torch.tensor([skip_batch], device = self.runner.device)
            torch.distributed.all_reduce(skip_batch_tensor, op = torch.distributed.ReduceOp.MAX)
            skip_batch = skip_batch_tensor.item()

        if skip_batch:
            self.runner.logger.error("该批次检测到NaN，所有进程跳过该batch")
        else:
            self.backward(self.scaler, model_output["loss"])
            self.register_model_output(model_output)

        self.runner.after_running_batch()

    @staticmethod
    def register_model_output(model_output):
        # 注册并报告所有包含"loss"的键值
        for key, value in model_output.items():
            if "loss" in key and isinstance(value, torch.Tensor):
                registry.register(f"metric.{key}", value.item())

                if registry.get("cfg.training.enable_tune", False):
                    import ray
                    ray.train.report(metrics = {key: value.item()})  # type: ignore

    def setup_scheduler(self):
        scheduler = None
        if not self.runner.is_deepspeed:
            scheduler_cls = registry.get_lr_scheduler_class(
                OmegaConf.select(self.runner.cfg, "scheduler.type", default = None)
            )
            if scheduler_cls:
                scheduler = scheduler_cls(
                    optimizer = self.runner.optimizer,
                    max_epoch = self.runner.epochs,
                    iters_per_epoch = int(get_batch_n(self.dataloader)),
                    **dict(self.runner.cfg.get("scheduler"))
                )
        else:
            print_log("如需要，请在deepspeed配置文件中指定scheduler", "current")
        return scheduler

    def maybe_autocast(self, enabled: bool = False):
        device_enable = self.runner.state.device != torch.device("cpu")
        if enabled and device_enable:
            return torch.autocast(device_type = "cuda", dtype = torch.float16)
        else:
            return contextlib.nullcontext()

    def backward(self, scaler, loss):
        if self.runner.is_deepspeed:
            self.runner.model.backward(loss)
            self.runner.model.step()
            return

        loss = loss / self.accumulation_steps if self.accumulation_steps > 1 else loss
        scaler.scale(loss).backward() if scaler else loss.backward()

        self.runner.state.accumulation_count += 1
        if self.runner.state.accumulation_count % self.accumulation_steps != 0:
            return

        if self.max_norm is not None:
            # 梯度裁剪
            # see https://docs.pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
            # Gradient accumulation部分，裁剪应该应用于原始的非缩放梯度
            if scaler:
                scaler.unscale_(self.runner.optimizer)  # 解除混合精度缩放
            torch.nn.utils.clip_grad_norm_(
                self.runner.model.parameters(),
                max_norm = self.max_norm,
                norm_type = 2
            )

        if scaler:
            scaler.step(self.runner.optimizer)
            scaler.update()
        else:
            if self.runner.optimizer is not None:
                self.runner.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step(self.runner.state.current_step)
        if self.runner.optimizer is not None:
            self.runner.optimizer.zero_grad()
        self.runner.state.accumulation_count = 0

    def resume(self):
        """
        跳过data_loader可能很耗时, 因此保守实现：恢复epoch，随机数，优化器，model，cfg
        """
        if self.runner.checkpoint_callback is None:
            raise RuntimeError("未设置checkpoint callback，无法复原训练状态！")
        start_epoch, start_batch = self.runner.checkpoint_callback \
            .load_checkpoint(self.runner.state.resume_from)  # type: ignore
        self._epoch = start_epoch
        self.runner.state.current_epoch = self._epoch
        print_log(f'恢复点 epoch: {start_epoch}', "current")

    def load(self):
        if hasattr(self.runner.model, "module"):
            # For DataParallel or DistributedDataParallel
            self.runner.model.module.load_checkpoint(self.runner.state.load_from)
        else:
            self.runner.model.load_checkpoint(self.runner.state.load_from)
