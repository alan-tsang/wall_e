import os
from ray import train, tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from omegaconf import OmegaConf


class Tuner:
    def __init__(self, train_func, tune_cfg):
        # 这里的train_func应该是原始的定义model，dataloader，然后输入到runner的整体（注意接受tune生成的cfg）
        self.train_func = train_func
        self.tune_cfg = tune_cfg

        self.__post_init__(tune_cfg)

    def __post_init__(self, cfg):
        search_space = self._build_param_space()
        scheduler = ASHAScheduler(
            # 这里是Ray Tune的调度器配置
            metric = OmegaConf.select(cfg, "tune.metric", default = "loss"),
            mode = OmegaConf.select(cfg, "tune.mode", default = "min"),
            max_t = OmegaConf.select(cfg, "tune.max_t", default = 100),
            grace_period = OmegaConf.select(cfg, "tune.grace_period", default = 10),
            reduction_factor = OmegaConf.select(cfg, "tune.reduction_factor", default = 2)
        )
        trainer = TorchTrainer(
            tune.with_parameters(self.train_func),
            scaling_config = ScalingConfig(
                num_workers = OmegaConf.select(cfg, "tune.num_workers", default = 1),
                use_gpu = OmegaConf.select(cfg, "tune.use_gpu", default = True),
                resources_per_worker = {"GPU": 1}
            )
        )
        path = os.path.join(
            os.getcwd(),
            OmegaConf.select(cfg, "tune.storage_path", default = "ray_results")
        )
        self.tuner = tune.Tuner(
            trainer,
            param_space = {"train_loop_config": search_space},
            tune_config = tune.TuneConfig(
                num_samples = OmegaConf.select(cfg, "tune.num_samples", default = 3),
                scheduler = scheduler,
                # metric = OmegaConf.select(cfg, "tune.metric", default = "loss"),
                # mode = OmegaConf.select(cfg, "tune.mode", default = "min"),
            ),
            run_config = train.RunConfig(
                name = OmegaConf.select(cfg, "tune.name", default = "tune_run"),
                storage_path = path,
            )
        )

    def tune(self):
        results = self.tuner.fit()
        print("Best config:", results.get_best_result().config)

    def _build_param_space(self):
        """构建Ray Tune搜索空间"""
        param_space = {}
        space_config = OmegaConf.select(self.tune_cfg, "tune.param_space", default = {})
        space_config = dict(space_config)

        for path, spec in space_config.items():
            if spec["type"] == "grid_search":
                param_space[path] = tune.grid_search(spec.get("values", []))
            elif spec["type"] == "uniform":
                param_space[path] = tune.uniform(spec["min"], spec["max"])
            elif spec["type"] == "choice":
                param_space[path] = tune.choice(spec["options"])
            elif spec["type"] == "loguniform":
                param_space[path] = tune.loguniform(spec["min"], spec["max"])
            elif spec["type"] == "normal":
                param_space[path] = tune.normal(spec["mean"], spec["stddev"])
            elif spec["type"] == "randint":
                param_space[path] = tune.randint(spec["low"], spec["high"])
        return param_space


def update_cfg_by_tune_params(base_cfg, tune_params):
    """动态更新配置参数"""
    tune_params = OmegaConf.create(tune_params)
    tune_params = tune_params.tune.param_space
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg))

    for key_path, value in tune_params.items():
        keys = key_path.split('.')
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    return cfg
