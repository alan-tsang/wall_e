import wandb
from omegaconf import OmegaConf


def load_cfg(path):
    base_cfg = OmegaConf.load(path)
    cli_cfg = OmegaConf.from_cli()
    ds_cfg = base_cfg.get("training.ds_config", None)
    ds_cfg = OmegaConf.load(ds_cfg) if ds_cfg else {}

    # is_sweep = base_cfg.training.get("is_sweep", False)
    # print("wandb sweep: ", is_sweep)
    runtime_cfg = {}
    # if is_sweep:
    #     """
    #     for wandb sweep, this can experiment with different hyperparameters
    #     """
    #     from ..dist.init import init_distributed_mode, is_main_process
    #     from ..dist.utils import barrier
    #     from ..dist.cmc import broadcast_object_list
    #
    #     init_distributed_mode()
    #     if is_main_process():
    #         wandb.init(
    #             project = base_cfg.wandb.wandb_project_name,
    #             name = base_cfg.run_name
    #         )
    #         runtime_cfg = wandb.config
    #     else:
    #         runtime_cfg = {}
    #
    #     broadcast_object_list([dict(runtime_cfg)], src = 0)
    #     barrier()

    cfg = OmegaConf.merge(base_cfg, dict(runtime_cfg), ds_cfg, cli_cfg)
    """
    if is_runtime, wandb will not init in the runner again, and as well, 
    the cfg should be updated to wandb.config
    """
    # if is_sweep:
    #     wandb.config.update(dict(cfg))

    return cfg
    # from .runner_config import RunnerConfig
    # return RunnerConfig(**cfg)

