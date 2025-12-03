from typing import Literal
import warnings
from .base_callback import BaseCallBack
from ..common.registry import registry


@registry.register_callback("early_stop")
class EarlyStopCallBack(BaseCallBack):
    """
    Early stopping callback to stop training when a monitored metric stops improving.
    
    Parameters:
        monitor (str): Metric to monitor (e.g., "val_loss", "train_loss").
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
        patience (int): Number of checks with no improvement after which training will be stopped.
                       - In 'epoch' mode: number of epochs
                       - In 'batch' mode: number of batches
        verbose (bool): Whether to print messages when training is stopped early.
        greater_is_better (bool): If True, higher metric values are better; if False, lower is better.
        mode (str): One of {"epoch", "batch"}. 
                   - "epoch": Check metric after each epoch
                   - "batch": Check metric after each batch
    """
    def __init__(
            self,
            runner,
            monitor: str,
            delta: float,
            patience: int,
            greater_is_better: bool,
            mode: Literal["epoch", "batch"],
            verbose: bool = True,
    ):
        super(EarlyStopCallBack, self).__init__(runner)
        self.monitor = monitor
        self.delta = delta
        self.patience = patience
        self.verbose = verbose
        self.greater_is_better = greater_is_better
        
        # 验证 mode 参数
        if mode not in ["epoch", "batch"]:
            raise ValueError(f"mode must be 'epoch' or 'batch', got '{mode}'")
        self.mode = mode
        
        # 状态变量
        self.wait = 0
        self.best_value = None
        self.best_step = 0  # 最佳值出现的步数
    
    def after_running_epoch(self):
        """Epoch 结束后检查（epoch 模式）"""
        if self.mode == "epoch":
            self._check_and_update()
    
    def after_running_batch(self):
        """Batch 结束后检查（batch 模式）"""
        if self.mode == "batch":
            self._check_and_update()
    
    def reset(self):
        """重置早停状态（可用于跨折验证等场景）"""
        self.wait = 0
        self.best_value = None
        self.best_step = 0
        if self.verbose:
            self.runner.logger.info("早停状态已重置")

    def _is_better(self, current: float, best: float) -> bool:
        """判断当前值是否优于最佳值"""
        if self.greater_is_better:
            return current > best + self.delta
        else:
            return current < best - self.delta
    
    def _get_current_step(self) -> int:
        """获取当前步数（epoch 或 batch）"""
        if self.mode == "epoch":
            return self.runner.state.current_epoch
        else:  # batch mode
            if not hasattr(self.runner.state, 'current_step'):
                raise AttributeError(
                    "batch 模式需要 runner.state.current_step 属性，但未找到。"
                    "请检查 Runner 是否正确维护了 current_step 状态。"
                )
            return self.runner.state.current_step
    
    def _get_metric_value(self):
        """获取监控的指标值"""
        return registry.get(f"metric.{self.monitor}")
    
    def _log_initialization(self, value: float):
        """记录初始化日志"""
        if self.verbose:
            self.runner.logger.info(
                f"早停初始化. 监控指标 '{self.monitor}' = {value:.4f}"
            )
    
    def _log_improvement(self, value: float):
        """记录指标改进日志"""
        if self.verbose:
            self.runner.logger.info(
                f"指标改进: {self.monitor} = {value:.4f} (当前最佳)"
            )
    
    def _check_and_update(self):
        """检查指标并更新早停状态"""
        current_value = self._get_metric_value()
        
        
        if current_value is None:
            return
        
        current_step = self._get_current_step()
        
        # 初始化最佳值
        if self.best_value is None:
            self.best_value = current_value
            self.best_step = current_step
            self._log_initialization(current_value)
            return
        
        # 检查是否有改进
        if self._is_better(current_value, self.best_value):
            self.best_value = current_value
            self.best_step = current_step
            self.wait = 0
            self._log_improvement(current_value)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._trigger_early_stop(current_step)
    
    def _trigger_early_stop(self, current_step: int):
        """触发早停"""
        self.runner.state.stop_training = True
        self._log_early_stop(current_step)
        
    def _log_early_stop(self, current_step: int):
        """记录早停日志"""
        if self.verbose:
            unit = "epoch" if self.mode == "epoch" else "batch"
            self.runner.logger.info(
                f"触发提前停止训练 ({unit}模式). "
                f"最佳 {self.monitor}: {self.best_value:.4f} 于 {unit} {self.best_step + 1}. "
                f"当前 {unit}: {current_step + 1}"
            )


__all__ = [
    'EarlyStopCallBack'
]