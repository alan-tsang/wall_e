import math
from datetime import datetime

from omegaconf import OmegaConf

from .base_callback import BaseCallBack
from ..common.registry import registry
from ..util.dl_util import get_batch_n


@registry.register_callback("progress")
class ProcessCallBack(BaseCallBack):
    """\
    打印训练进度，以及 registry 中 metric.* 的值。

    说明：
    - 不再依赖 config 里手动指定 metric key；进度条会从 registry.metric.* 自动发现。
    - greater_is_better 的方向信息来自 registry.metric_meta.*，由 BaseMetric.monitor_metrics 提供。
      （当 meta 缺失时，默认按“越小越好”处理）
    """

    def __init__(self, runner):
        super().__init__(runner)

        self.logger = runner.logger
        self.every_n_epoch = OmegaConf.select(
            self.runner.cfg,
            "training.progress_every_n_epochs",
            default=1,
        )
        self.every_n_batches = OmegaConf.select(
            self.runner.cfg,
            "training.progress_every_n_batches",
            default=1,
        )

        # Track already attached meters so we can dynamically add more later
        self._meter_by_key: dict[str, AverageMeter] = {}

        self.__post_init__()

    def __post_init__(self):
        self.progress_meter = ProgressMeter(
            self.runner.epochs,
            int(get_batch_n(self.runner.train_data_loader)),
            meters=[],
        )
        self._refresh_meters_from_registry()

    def _discover_metric_keys(self) -> list[str]:
        """Discover available metric keys under registry.state.metric."""
        metric_state = registry.get("metric", default=None, no_warning=True)
        if metric_state is None:
            return []
        keys = [str(k) for k in metric_state.keys()]

        # stable order: loss* first, then alphabetical
        loss_keys = sorted([k for k in keys if "loss" in k.lower()])
        other_keys = sorted([k for k in keys if k not in loss_keys])
        return loss_keys + other_keys

    def _refresh_meters_from_registry(self) -> None:
        """Ensure every discovered metric key has a corresponding meter."""
        keys = self._discover_metric_keys()

        for key in keys:
            if key in self._meter_by_key:
                continue
            # infer direction from meta registered by Evaluator/BaseMetric
            greater_is_better = bool(
                registry.get(f"metric_meta.{key}", default=False, no_warning=True)
            )
            meter = AverageMeter(
                monitor=key,
                greater_is_better=greater_is_better,
            )
            self._meter_by_key[key] = meter
            self.progress_meter.meters.append(meter)

    def after_running_batch(self):
        # Meters might be empty during early iterations; try to discover on the fly.
        self._refresh_meters_from_registry()

        for meter in self.progress_meter.meters:
            value = registry.get(f"metric.{meter.monitor}", default=None, no_warning=True)
            if value is None:
                meter.update(float("nan"))
            else:
                meter.update(float(value))

        current_epoch = self.runner.state.current_epoch
        current_step = self.runner.state.current_step
        if (
            self.every_n_batches
            and current_step % self.every_n_batches == 0
            and self.every_n_epoch
            and current_epoch % self.every_n_epoch == 0
        ):
            self.progress_meter.display(current_epoch, current_step, self.logger)


class ProgressMeter:
    def __init__(self, epoch_n, batch_n, meters, prefix=""):
        self.batch_n = batch_n
        self.epoch_n = epoch_n
        self.epoch_fmtstr = self._get_progress_fmtstr(epoch_n)
        self.batch_fmtstr = self._get_progress_fmtstr(batch_n)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch_i, batch_i, logger):
        current_time = datetime.now()
        if not hasattr(self, "start_time"):
            self.start_time = current_time

        elapsed_time = (current_time - self.start_time).total_seconds()

        percent = math.ceil(100.0 * (batch_i) / self.batch_n) if self.batch_n else 100
        speed = 0.0
        if batch_i > 0 and elapsed_time > 0:
            speed = batch_i / elapsed_time

        progress_length = 15
        filled_length = (
            min(math.ceil(progress_length * (batch_i + 1) / self.batch_n), progress_length)
            if self.batch_n
            else progress_length
        )
        bar = "━" * filled_length + "-" * (progress_length - filled_length)

        remaining_time = 0.0
        if batch_i > 0 and elapsed_time > 0 and self.batch_n:
            remaining_time = (self.batch_n - batch_i) * elapsed_time / batch_i

        entries = [self.prefix]
        entries += "epoch " + self.epoch_fmtstr.format(epoch_i) + " "
        entries += f"| "
        entries += self.batch_fmtstr.format(batch_i)
        entries += f"  {bar} "
        entries += f"〔 {elapsed_time:.2f}s<{remaining_time:.2f}s, "
        entries += f"{speed:.2f}it/s 〕"
        entries += [str(meter) for meter in self.meters]
        logger.just_print("".join(entries))


    @staticmethod
    def _get_progress_fmtstr(n):
        num_digits = len(str(n // 1))
        fmt = "{:" + str(num_digits) + "d}"

        # return "[" + fmt + f"/{n}" + "]"
        return fmt + f"/{n}"


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, monitor, avg_print=False, greater_is_better=False, fmt=".3f"):
        self.monitor = monitor
        self.avg_print = avg_print
        self.greater_is_better = greater_is_better
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.best = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.best = None
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        if math.isnan(self.val):
            self.best = self.val
            self.count += n
            return

        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.best is None or math.isnan(self.best):
            self.best = self.val
        else:
            self.best = max(self.best, self.val) if self.greater_is_better else min(self.best, self.val)


    def __str__(self):
        arrow = "↑" if self.greater_is_better else "⬇"
        if self.avg_print:
            fmtstr = (
                f" | {arrow} {self.monitor.lower()}: {format(self.val, self.fmt)} "
                f"(best: {format(self.best, self.fmt)})(avg: {format(self.avg, self.fmt)}) "
            )
        else:
            fmtstr = f" | {arrow} {self.monitor.lower()}: {format(self.val, self.fmt)} (best: {format(self.best, self.fmt)})"
        return fmtstr


if __name__ == '__main__':
    # 示例应用
    num_batches = 100
    meters = [AverageMeter(f"x{_}") for _ in range(2)]  # 两个监测器
    progress_meter = ProgressMeter(10 , num_batches, meters)
    for epoch_i in range(1, 11):
        for batch_i in range(1, num_batches + 1):
            # 模拟计算损失
            loss = batch_i * 0.1  # 假设的损失值
            meters[0].update(loss)
            meters[1].update(loss / 2)  # 假设的其他指标

            # 每10个批次显示一次进度
            if batch_i % 10 == 0:
                progress_meter.display(epoch_i, batch_i)


__all__ = [
    'ProcessCallBack', 'ProgressMeter', 'AverageMeter'
]
# def after_running_epoch(self):
#     for meter in self.progress_meter.meters:
#         if registry.get(f"metric.{meter.monitor})"):
#             meter.update(registry.get(f"metric.{meter.monitor}"))
#         else:
#             meter.update(float("nan"))
#
#     current_epoch = registry.get("current_epoch")
#     current_batch = registry.get("current_batch")
#     if self.every_n_epoch and current_epoch % self.every_n_epoch == 0:
#         self.progress_meter.display(current_epoch, self.progress_meter.batch_n - 1, self.logger)
