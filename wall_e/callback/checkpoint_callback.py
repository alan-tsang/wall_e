import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf

from .base_callback import BaseCallBack
from ..logging.logger import Logger
from ..common.registry import registry
from ..common.util import now
from ..dist import barrier


@registry.register_callback("checkpoint")
class CheckpointCallback(BaseCallBack):
    def __init__(
            self,
            runner
    ):
        super().__init__(runner)

        self.logger = runner.logger
        cfg = self.runner.cfg
        self.save_enabled = self.runner.is_main_process or self.runner.is_deepspeed
        self.begin_epoch = OmegaConf.select(cfg, "pt.begin_epoch", default = 1)
        self.epoch_interval = OmegaConf.select(cfg, "pt.epoch_interval", default = 1)
        self.begin_batch = OmegaConf.select(cfg, "pt.begin_batch", default = 1)
        self.batch_interval = OmegaConf.select(cfg, "pt.batch_interval", default = None)

        if self.runner.is_main_process:
            self.folder = self.generate_save_folder(
                base_folder = OmegaConf.select(cfg, "run_dir", default = './run'),
                prefix = OmegaConf.select(cfg, "run_name", default = 'default')
            )
        else:
            self.folder = None
        # 如果是deepspeed，广播保存目录
        if self.runner.is_deepspeed:
            _ = [self.folder]
            torch.distributed.broadcast_object_list(_, src = 0)
            self.folder = _[0]
        self.runner.logger.info(f"检查点保存位置：{self.folder}")

        # TOPK相关的保存
        monitor = list(
            OmegaConf.select(cfg, "pt.best_monitor", default = {"loss": False}).items()
        )[0]
        self.topk = OmegaConf.select(cfg, "pt.topk", default = 3)
        self.monitor = monitor[0]
        self.monitor_greater_is_better = monitor[1]
        self.topk_models: List[Dict] = []

        self.__post_init__()

    def __post_init__(self):
        if self.folder:
            os.makedirs(self.folder, exist_ok = True)
        self.validate_init_params(self.topk, self.monitor)

    @staticmethod
    def validate_init_params(topk: int, monitor: Optional[str]):
        if topk > 0 and not monitor:
            raise ValueError("当topk>0时, 必须指定监控指标(monitor)")

    @staticmethod
    def generate_save_folder(base_folder: str, prefix: str) -> str:
        """生成带时间戳的子过程检查点保存文件名（不含后缀）"""
        timestamp = registry.get("cfg.run_timestamp")
        folder = os.path.join(base_folder, prefix, timestamp, "checkpoint")
        return folder

    def _get_model_state(self):
        return self.runner.model.module.state_dict() if hasattr(self.runner.model, "module") \
            else self.runner.model.state_dict()

    def _get_optimizer_state(self):
        if self.runner.is_deepspeed:
            # DeepSpeed自行管理
            return None
        return self.runner.optimizer.state_dict()

    def _get_scheduler_state(self):
        return self.runner.scheduler.state_dict() if self.runner.scheduler else None

    def save_checkpoint(self, name: str):
        """智能保存检查点"""
        if self.runner.is_deepspeed:
            return self._save_deepspeed_checkpoint(name)
        else:
            return self._save_standard_checkpoint(name)

    def _save_deepspeed_checkpoint(self, name: str):
        """DeepSpeed专用保存"""
        from deepspeed.utils import logger as ds_logger
        client_state = {
            "epoch": self.runner.state.current_epoch,
            "batch": self.runner.state.current_batch,
            "rng_state": torch.get_rng_state(),
            "cfg": self.runner.cfg
        }
        save_path = os.path.join(self.folder, name)
        self.runner.model.save_checkpoint(
            save_dir = self.folder,
            # 子目录
            tag = name,
            client_state = client_state,
            save_latest = True
        )
        # deepspeed自带这样的日志,但不会写入到自定义日志中
        ds_logger.info(f"DeepSpeed检查点保存至: {save_path}")
        return save_path

    def _save_standard_checkpoint(self, name: str):
        """保存标准PyTorch检查点（非DeepSpeed环境）"""
        # 确保仅在主进程执行保存操作
        if not self.runner.is_main_process:
            return

        # 获取完整训练状态
        checkpoint = {
            "model": self._get_model_state(),
            "optimizer": self._get_optimizer_state(),
            "scheduler": self._get_scheduler_state(),
            "epoch": self.runner.state.current_epoch,
            "batch": self.runner.state.current_batch,
            "rng_state": torch.get_rng_state(),
            "cfg": dict(self.runner.cfg),
        }

        # 保存到文件
        if not name.endswith(".pt"):
            name = name + ".pt"
        save_path = os.path.join(self.folder, name)
        torch.save(checkpoint, save_path)
        self.logger.info(f"检查点保存至: {save_path}")
        return save_path

    # 各生命周期回调方法
    def before_train(self):
        """训练开始前保存初始模型"""
        self.save_checkpoint("initial")
        self.logger.info(f"初始模型保存成功: initial")

    def after_train(self):
        """训练结束后保存最终模型"""
        self.save_checkpoint("final")
        self.logger.info(f"最终模型保存成功: final")

    def after_running_epoch(self):
        current_epoch = self.runner.state.current_epoch
        """周期结束处理"""
        epoch_gap = current_epoch - self.begin_epoch
        if self.save_enabled and epoch_gap >= 0 and epoch_gap % self.epoch_interval == 0:
            self.save_checkpoint(f"epoch_{current_epoch}")
            if not self.runner.is_deepspeed:
                self.save_checkpoint(f"latest")

            # =====================
            if self.topk > 0 and self.monitor is not None:
                self.handle_topk(current_epoch, None)

    def after_running_batch(self):
        current_epoch = self.runner.state.current_epoch
        current_step = self.runner.state.current_step
        """批次结束处理"""
        batch_gap = current_step - self.begin_batch
        if self.save_enabled and batch_gap >= 0 and batch_gap % self.batch_interval == 0:
            self.save_checkpoint(f"epoch_{current_epoch}-batch_{current_step}")
            if not self.runner.is_deepspeed:
                self.save_checkpoint(f"latest")

            # ========================
            if self.topk > 0 and self.monitor is not None:
                self.handle_topk(current_epoch, current_step)

    def on_exception(self, exception: Exception):
        """异常处理"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exception_{type(exception).__name__}_{ts}"
        self.save_checkpoint(filename)
        self.logger.error(f"异常发生时保存模型: {filename}")

    def load_checkpoint(self, path: str):
        """智能加载检查点"""
        if self.runner.is_deepspeed:
            return self._load_deepspeed_checkpoint(path)
        else:
            return self._load_standard_checkpoint(path)

    def _load_deepspeed_checkpoint(self, path: str):
        tag = os.path.split('/')[-1]
        if tag == "latest":
            latest_file = os.path.join(path, "latest")
            with open(latest_file, 'r') as f:
                tag = f.read().strip()

        load_path, client_state = self.runner.model.load_checkpoint(
            load_dir = path,
            tag = tag,
            load_module_strict = True,
            load_optimizer_states = True,
            load_lr_scheduler_states = True
        )

        # 恢复附加状态
        if client_state:
            torch.set_rng_state(client_state["rng_state"])
            self.runner.cfg = client_state["cfg"]
            registry.register("cfg", client_state["cfg"])

        return client_state.get("epoch", 0), client_state.get("batch", 0)

    def _load_standard_checkpoint(self, path: str) -> tuple[int, int]:
        """加载标准PyTorch检查点（非DeepSpeed环境）"""
        checkpoint = torch.load(path, map_location = "cpu")

        # 1. 加载模型状态
        model = self.runner.model
        if hasattr(model, "module"):  # 分布式训练包装
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        # 2. 加载优化器状态
        if checkpoint["optimizer"] is not None:
            self.runner.optimizer.load_state_dict(checkpoint["optimizer"])

        # 3. 加载学习率调度器状态
        if checkpoint["scheduler"] is not None and self.runner.scheduler is not None:
            self.runner.scheduler.load_state_dict(checkpoint["scheduler"])

        # 4. 恢复随机状态
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])

        # 5. 恢复配置
        if "cfg" in checkpoint:
            self.runner.cfg = checkpoint["cfg"]
            registry.register("cfg", checkpoint["cfg"])

        # 6. 返回恢复的epoch和batch位置
        return checkpoint.get("epoch", 0), checkpoint.get("batch", 0)

    def handle_topk(self, epoch: int, batch: Optional[int]):
        """处理TopK模型保存逻辑"""
        if (value := registry.get(f"metric.{self.monitor}")) is None:
            self.logger.warning(f"监控指标 '{self.monitor}' 不存在")
            return

        self._update_topk_models(value, epoch, batch)

    def _update_topk_models(
            self, current_value: float, epoch: int, batch: Optional[int]
    ):
        """维护TopK模型列表并更新最佳模型"""
        filename = self._generate_topk_filename(epoch, batch, current_value)
        self._add_to_topk(filename, current_value)
        self._prune_topk_list()
        self._update_best_model()

    def _generate_topk_filename(
            self, epoch: int, batch: Optional[int], value: float
    ) -> str:
        """
        生成topk的文件名（无后缀）
        """
        base = f"epoch_{epoch}"
        if batch is not None:
            base += f"-batch_{batch}"
        filename = f"{base}-{self.monitor}_{value:.4f}"
        return filename

    def _add_to_topk(self, filename: str, value: float):
        """尝试添加新模型到TopK列表"""
        if len(self.topk_models) < self.topk or self._is_better_than_worst(value):
            save_path = self.save_checkpoint(filename)
            self.topk_models.append(
                {"value": value, "path": save_path}
            )
            self.topk_models.sort(key = lambda x: x["value"], reverse = self.monitor_greater_is_better)

    def _is_better_than_worst(self, value: float) -> bool:
        """判断当前值是否优于最差TopK值"""
        if not self.topk_models:
            return False
        worst = self.topk_models[-1]["value"]
        return value > worst if self.monitor_greater_is_better else value < worst

    def _prune_topk_list(self):
        while len(self.topk_models) > self.topk:
            removed = self.topk_models.pop()
            path_to_remove = removed["path"]

            if not os.path.exists(path_to_remove):
                self.logger.warning(f"路径不存在，跳过删除: {path_to_remove}")
                continue
            try:
                if os.path.isdir(path_to_remove):
                    shutil.rmtree(path_to_remove)
                    self.logger.info(f"删除目录: {path_to_remove}")
                else:
                    os.remove(path_to_remove)
                    self.logger.info(f"删除文件: {os.path.basename(removed['path'])}")
            except Exception as e:
                self.logger.error(f"删除失败 {path_to_remove}: {str(e)}")

    def _update_best_model(self):
        if not self.topk_models:
            return
        best = self.topk_models[0]
        best_source = best["path"]
        if self.runner.is_deepspeed:
            if self.runner.is_main_process:
                best_dest = os.path.join(self.folder, "best")
                if os.path.exists(best_dest):
                    shutil.rmtree(best_dest)
                shutil.copytree(best_source, best_dest)
        else:
            best_dest = os.path.join(self.folder, "best.pt")
            shutil.copyfile(best_source, best_dest)
        if self.runner.is_main_process:
            self.logger.info(f"更新最佳模型: {self.monitor}={best['value']:.4f} -> {best_dest}")


__all__ = ["CheckpointCallback"]
