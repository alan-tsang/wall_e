import torch
from omegaconf import OmegaConf

from ..util.dl_util import get_model_info
from ..common.util import now


class RunnerState:
    def __init__(self, runner):
        self.runner = runner
        self.cfg = runner.cfg
        self.__post_init__(runner)
        self.__post_init_runtime()

    def __post_init__(self, runner):
        cfg = runner.cfg
        self.run_timestamp = now()
        cfg.run_timestamp = self.run_timestamp
        self.ds_config = OmegaConf.select(cfg, "training.ds_config", default = None)
        self.epochs = runner.epochs if runner.epochs is not None else 1
        self.resume_from = OmegaConf.select(cfg, "training.resume_from", default = None)
        self.load_from = OmegaConf.select(cfg, "training.load_from", default = None)
        self.activation_checkpoint = OmegaConf.select(cfg, "training.activation_checkpoint", default = None)

        self.train_size = len(runner.train_data_loader.dataset) \
            if runner.train_data_loader else 0
        self.valid_size = len(runner.valid_data_loader.dataset) \
            if runner.valid_data_loader else 0
        self.test_size = len(runner.test_data_loader.dataset) \
            if runner.test_data_loader else 0

    def __post_init_runtime(self):
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0

        self.start_epoch = 0
        self.start_batch = 0

        self.accumulation_count = 0
        self.stop_training = False
        self.is_main_process = True
        self.device = torch.device("cpu")
        self.start_msg = "Runner is starting..."
        self.model_info = get_model_info(self.runner.model)

        # 使用registry做了全局变量
        # self.metric = {}
