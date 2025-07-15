import wandb
from omegaconf import OmegaConf

from .base_callback import BaseCallBack
from ..common.registry import registry


@registry.register_callback("wandb")
class WandbCallback(BaseCallBack):
    def __init__(self, runner):
        super().__init__(runner)
        if self.runner.is_main_process and wandb.run is None:
            cfg = self.runner.cfg
            wandb.init(
                project=OmegaConf.select(cfg,"wandb.proj_name", default = 'default'),
                name=OmegaConf.select(cfg,"run_name", default = 'default'),
                notes = cfg.get("run_description", ""),
                mode="offline" if OmegaConf.select(cfg,"wandb.offline", default = False) \
                    else "online",
                config = dict(cfg),
                dir = OmegaConf.select(cfg,"wandb.dir", default = './'),
                save_code = True,
                tags = OmegaConf.select(cfg,"wandb.dir", default = None)
            )

    def after_running_batch(self):
        if wandb.run is not None:
            metrics: dict = registry.get("metric")
            if metrics is not None:
                for key, value in metrics.items():
                    wandb.log({key: value})

    # def after_running_epoch(self):
    #     if wandb.run is not None:
    #         for key, value in registry.get("metric").items():
    #             wandb.log({key: value})
