""" Runner for training, validation and testing.

This module provides the core training execution engine for AI models.
It handles distributed training, model optimization, logging, and callback management.
"""
import traceback
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from omegaconf import omegaconf, OmegaConf
from torch.utils.data.distributed import DistributedSampler

from .base_runner import RunnerBase
from .state import RunnerState
from .. import BaseModel, Evaluator
from ..callback import (CheckpointCallback, EpochSummaryCallBack,
                        ProcessCallBack, WandbCallback)
from ..callback.base_callback import BaseCallBack
from ..common.registry import registry
from ..common.util import better_dict_4_print
from ..dist.init import is_main_process
from ..logging.logger import Logger



class Runner(RunnerBase):
    """
    AI 模型训练核心执行器

    ██████╗ ██╗███╗   ██╗███╗   ██╗███████╗██████╗
    ██╔══██╗██║████╗  ██║████╗  ██║██╔════╝██╔══██╗
    ██████╔╝██║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝
    ██╔══██╗██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗
    ██║  ██║██║██║ ╚████║██║ ╚████║███████╗██║  ██║
    ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝

    典型用法：
    custom your yaml config file based on the cfg in example,
    >>> runner = Runner(
            model=your_model,
            train_data_loader=train_loader,
            ...
        )
    >>> runner.fit()

    扩展点：
    - 继承 TrainingLoop 类实现自定义训练逻辑
    - 通过 callback_manager 注册自定义回调
    """
    def __init__(
            self,
            model: BaseModel,
            train_data_loader: torch.utils.data.DataLoader,
            valid_data_loader: torch.utils.data.DataLoader = None,
            test_data_loader: torch.utils.data.DataLoader = None,
            train_loop = None,
            valid_loop = None,
            test_loop = None,
            train_evaluator: Evaluator = None,
            valid_evaluator: Evaluator = None,
            test_evaluator: Evaluator = None,
            epochs: int = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            cfg: Union[dict, omegaconf.DictConfig] = None,

            *args,
            **kwargs,
    ):
        """
        初始化训练执行器
        
        Args:
            model (BaseModel): 要训练的模型
            train_data_loader (torch.utils.data.DataLoader): 训练数据加载器
            valid_data_loader (torch.utils.data.DataLoader, optional): 验证数据加载器
            test_data_loader (torch.utils.data.DataLoader, optional): 测试数据加载器
            train_loop: 自定义训练循环，如果为None则使用默认TrainLoop
            valid_loop: 自定义验证循环，如果为None则使用默认ValidLoop
            test_loop: 自定义测试循环，如果为None则使用默认TestLoop
            train_evaluator (Evaluator, optional): 训练评估器
            valid_evaluator (Evaluator, optional): 验证评估器
            test_evaluator (Evaluator, optional): 测试评估器
            epochs (int, optional): 训练轮数
            optimizer (torch.optim.Optimizer, optional): 优化器，如果为None则使用默认AdamW
            cfg (Union[dict, omegaconf.DictConfig], optional): 配置字典或OmegaConf配置对象
        """
        super().__init__()
        self.model = model

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.train_loop = train_loop
        self.valid_loop = valid_loop
        self.test_loop = test_loop

        self.train_evaluator = train_evaluator
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator

        self.epochs = epochs
        # NOTE: 启用deepspeed时，optimizer可以为空
        self.optimizer = optimizer
        self.cfg = cfg

        self.state = RunnerState(self)
        registry.register("cfg", cfg)

        self.__post_init__()

    def __post_init__(self):
        """
        初始化后的设置步骤，按顺序执行：
        1. 设置启动策略（单机/分布式）
        2. 设置日志记录器
        3. 设置模型和优化器
        4. 设置训练/验证/测试循环
        5. 设置回调函数
        6. 设置评估器
        """
        self.setup_launch_strategy()
        self.logger = self.setup_logger()
        self.setup_model_optimizer()
        self.setup_loop()
        self.setup_callbacks()
        self.setup_evaluator()

    @property
    def is_deepspeed(self):
        return self.state.ds_config is not None

    @property
    def device(self):
        return self.state.device

    @property
    def is_main_process(self):
        return self.state.is_main_process

    @property
    def current_epoch(self):
        return self.state.current_epoch

    @property
    def current_step(self):
        return self.state.current_step

    def fit(self, *args, **kwargs):
        """
        执行完整的训练流程
        
        包括训练前的准备、训练循环的执行、异常处理和训练后的清理工作。
        如果发生异常，会记录详细的错误信息并调用异常处理回调（默认保存训练状态）。
        """
        try:
            self.before_fit()
            # valid_loop、test_loop 服务于train_loop，被内置管理
            self.train_loop.run()
        except BaseException as e:
            # 获取完整的异常堆栈信息
            error_trace = traceback.format_exc()
            self.logger.critical(
                f"训练时发生严重异常：{e.__class__.__name__}\n"
                f"异常信息：{e}\n"
                f"完整报错路径：\n{error_trace}"
            )
            self.on_exception(e)
            raise
        finally:
            self.after_fit()

    def train(self):
        self.train_loop.run()

    def valid(self):
        self.logger.info("开始验证...")
        self.valid_loop.run()

    def test(self):
        self.logger.info("开始测试...")
        return self.test_loop.run()

    def before_fit(self):
        super().before_fit()
        self.log_initial_info()

    def log_initial_info(self):
        """
        记录初始信息，包括：
        - 启动方式（单机/分布式/DeepSpeed）
        - 设备信息
        - 配置信息
        - 模型信息
        """
        self.logger.info(f"启动方式：{self.state.start_msg}")
        self.logger.info(f"启动设备：{self.state.device}")
        self.logger.info(f"当前运行配置：\n{better_dict_4_print(self.cfg)}")
        self.logger.info(f"当前运行模型：{self.model.__class__.__name__}")
        if OmegaConf.select(self.cfg, "training.print_model", default = True):
            # support distributed print, means print only once in the main process
            self.logger.just_print(self.state.model_info)

    def after_fit(self):
        super().after_fit()

    def cleanup_resources(self):
        self.logger.close_file_handler()
        if dist.is_initialized():
            dist.destroy_process_group()

    def setup_loop(self):
        """
        设置训练、验证和测试循环
        
        如果未提供自定义循环，则创建默认的循环实例：
        - TrainLoop: 处理训练逻辑，包括验证和测试的调度
        - ValidLoop: 处理验证逻辑
        - TestLoop: 处理测试逻辑
        """
        if self.train_loop is None and self.train_data_loader is not None:
            from .loop.train_loop import TrainLoop
            train_loop_cfg = self.cfg.training
            self.train_loop = TrainLoop(
                runner = self,
                dataloader = self.train_data_loader,
                max_epochs = self.epochs,
                valid_begin_epoch = train_loop_cfg.get("valid_begin_epoch", 1),
                valid_interval_epoch = train_loop_cfg.get("valid_interval_epoch", 1),
                valid_begin_iter = train_loop_cfg.get("valid_begin_iter", 4000),
                valid_interval_iter = train_loop_cfg.get("valid_interval_iter", 1000),
                test_begin_epoch = train_loop_cfg.get("test_begin_epoch", 1),
                test_interval_epoch = train_loop_cfg.get("test_interval_epoch", 1),
                test_begin_iter = train_loop_cfg.get("test_begin_iter", 4000),
                test_interval_iter = train_loop_cfg.get("test_interval_iter", 1000),
            )
        if self.valid_loop is None and self.valid_data_loader is not None:
            from .loop.valid_loop import ValidLoop
            self.valid_loop = ValidLoop(
                runner = self,
                dataloader = self.valid_data_loader,
                evaluator = self.valid_evaluator
            )
        if self.test_loop is None and self.test_data_loader is not None:
            from .loop.test_loop import TestLoop
            self.test_loop = TestLoop(
                runner = self,
                dataloader = self.test_data_loader,
                evaluator = self.test_evaluator
            )

    def setup_logger(self):
        """
        设置日志记录器
        
        从配置中读取日志相关参数：
        - level: 主进程日志级别
        - rank_level: 分布式训练时从进程日志级别
        - to_file: 是否输出到文件
        - folder: 日志文件目录
        - run_name: 运行名称
        """
        return Logger.get_instance(
            "runner",
            level = OmegaConf.select(self.cfg, "log.level", default = 'INFO'),
            rank_level = OmegaConf.select(self.cfg, "log.rank_level", default = "WARNING"),
            to_file = OmegaConf.select(self.cfg, "log.to_file", default = True),
            folder = OmegaConf.select(self.cfg, "log.folder", default = './logs'),
            run_name = OmegaConf.select(self.cfg, "run_name", default = 'default')
        )

    def setup_launch_strategy(self):
        """
        根据设备数量、类型，自动判断单机、分布式并行、CPU
        
        启动方式：
        - 单机、CPU: python ./train.py
        - 分布式: python -m torch.distributed.launch --nproc_per_node=4 ./train.py
        
        该函数兼容ray的分布式环境处理
        """
        if torch.cuda.is_available():
            # if dist.is_available() and not dist.is_initialized():
            from ..dist.init import init_distributed_mode
            if not dist.is_initialized():
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1 or self.is_deepspeed:
                    is_successful = init_distributed_mode()
                    if is_successful is False:
                        raise Exception("分布式环境初始化失败！")

            if dist.is_initialized():
                device = torch.device(f'cuda:{dist.get_rank()}')
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.state.device = device
        self.state.is_main_process = is_main_process()

    def setup_model_optimizer(self):
        """
        设置模型和优化器
        
        包括：
        1. 模型检查点重载
        2. 激活值重算设置（用于节省显存）
        3. 模型设备迁移
        4. 分布式训练包装（DDP或DeepSpeed）
        5. 默认优化器创建
        """
        # 检查点重载
        if self.state.load_from:
            self.logger.info(f"从检查点 {self.state.load_from} 加载权重...")
            self.model.load_checkpoint(self.state.load_from)

        # 激活值重算
        if (
            modules := OmegaConf.select(
                self.cfg, "training.activation_checkpoint",
                default = None
            )
        ) is not None:
            self.logger.info(f"启用激活检查点: {modules}")
            from .activation_checkpointing import turn_on_activation_checkpointing
            turn_on_activation_checkpointing(self.model, modules)

        # 分布式模型重载
        device = self.device
        if device.type == "cpu":
            self.state.start_msg = "使用CPU启动中..."
            return

        self.model = self.model.to(device)
        if dist.is_initialized():
            if self.is_deepspeed:
                import deepspeed
                self.model, self.optimizer, _, _ = deepspeed.initialize(
                    model = self.model,
                    optimizer = self.optimizer,
                    model_parameters = self.model.parameters(),
                    config = self.state.ds_config,
                )
                self.state.start_msg = "使用deepspeed启动中..."
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids = [dist.get_rank()],
                    output_device = dist.get_rank()
                )
                self.state.start_msg = "使用分布式训练启动中..."
        else:
            self.state.start_msg = "使用单卡启动中..."

        # 如果没有设置优化器，创建默认优化器
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr = self.cfg.optimizer.lr,
                weight_decay = self.cfg.optimizer.weight_decay
            )


    def setup_callbacks(self):
        """
        设置回调函数
        
        默认包括：
        - EpochSummaryCallBack: 轮次总结回调，总结全局注册中metrics的内容，
        即registry.get('metric')中的内容：runner运行时自动注册含loss关键字的train_step输出，
        以及evaluator执行的输出；这些内容也会被自动注册wandb（如果开启）
        - ProcessCallBack: 训练进度回调
        - WandbCallback: Weights & Biases日志回调（可选）
        - CheckpointCallback: 训练状态保存回调
        """
        epoch_summary_callback = EpochSummaryCallBack(self)
        progress_callback = ProcessCallBack(self)
        wandb_callback = None
        if OmegaConf.select(self.cfg, "wandb.enable", default = True):
            try:
                import wandb
            except ImportError:
                raise ImportError(
                    "未安装wandb，无法使用wandb callback。"
                    "请运行以下命令安装：\n"
                    "pip install wandb"
                )
            else:
                wandb_callback = WandbCallback(self)

        if OmegaConf.select(self.cfg, "pt.enable", default = True) \
                or self.state.resume_from:
            checkpoint_callback = CheckpointCallback(self)
        else:
            checkpoint_callback = None

        self.checkpoint_callback = checkpoint_callback

        callbacks = [
            progress_callback,
            checkpoint_callback,
            wandb_callback,
            epoch_summary_callback,
        ]
        callbacks.extend(self.callbacks)
        self.register_callbacks(callbacks)

    def extend_callbacks(self, callbacks: List[BaseCallBack]):
        """
        扩展回调函数列表，该API用于用户自定义扩展callback
        
        Args:
            callbacks (List[BaseCallBack]): 要添加的回调函数列表
        """
        self.callbacks.extend(callbacks)

    def setup_evaluator(self):
        """
        设置评估器
        
        为训练、验证和测试评估器设置状态信息
        """
        if self.train_evaluator is not None:
            self.train_evaluator.setup_state(self.state)
        if self.valid_evaluator is not None:
            self.valid_evaluator.setup_state(self.state)
        if self.test_evaluator is not None:
            self.test_evaluator.setup_state(self.state)

    @staticmethod
    def wrap_dataloader(data_loader, shuffle):
        """
        在分布式训练中，使用DistributedSampler来确保不同进程处理不同的数据子集,
        包装数据加载器以支持分布式训练;
        
        Args:
            data_loader (torch.utils.data.DataLoader): 原始数据加载器
            shuffle (bool): 是否打乱数据
            
        Returns:
            torch.utils.data.DataLoader: 包装后的数据加载器
            
        """
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                data_loader.dataset,
                num_replicas = dist.get_world_size(),
                rank = dist.get_rank(),
                shuffle = shuffle
            )
            return torch.utils.data.DataLoader(
                data_loader.dataset,
                batch_size = data_loader.batch_size,
                sampler = sampler,
                collate_fn = data_loader.collate_fn,
                shuffle = False,
                num_workers = data_loader.num_workers,
                pin_memory = True,
                drop_last = data_loader.drop_last
            )
        else:
            return data_loader

    # @main_process
    # def wandb_log(self, log_dict):
    #     if self.cfg.get("wandb.wandb_enable", False):
    #         try:
    #             import wandb
    #         except ImportError:
    #             warnings.warn("未安装wandb，请使用pip install wandb安装.")
    #         else:
    #             if wandb.run is not None:
    #                 wandb.log(log_dict)
    #                 self.logger.info(f"wandb记录：{log_dict}")
