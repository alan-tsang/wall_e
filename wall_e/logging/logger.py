import os
import sys
import logging
from typing import Optional

import torch

from ..dist import master_only
from ..dist import is_main_process, get_rank
from ..common.mixin import ManagerMixin
from ..common.util import now


class LoggerFormatter(logging.Formatter):
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        # 在日志格式中添加进程ID和进程排名信息
        fmt = fmt or ('[%(asctime)s] | [PID:%(process)d RANK:%(rank)s] | [%(levelname)s] | '
                      '[%(filename)s:%(funcName)s():%(lineno)d] %(message)s')
        super().__init__(fmt, datefmt)

    def format(self, record):
        # 添加进程排名信息到日志记录
        if not hasattr(record, 'rank'):
            record.rank = get_rank() if torch.distributed.is_initialized() else 0
        return super().format(record)


class Logger(ManagerMixin):
    """
    分布式日志记录器，区分主进程和子进程
    """
    LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(
            self,
            name: str,
            level: str,
            rank_level: Optional[str] = None,  # 非主进程的日志级别
            to_file: bool = True,
            folder: str = "./logs",
            run_name: str = "",
            **kwargs
    ):
        super().__init__(name, **kwargs)
        self.name = name
        self.is_master = is_main_process()
        self.to_file = to_file

        # 设置日志级别：主进程使用level，其他进程使用rank_level
        if rank_level is None:
            rank_level = level
        current_level = level if self.is_master else rank_level

        self.logger = logging.getLogger(f"{name}-rank{get_rank()}")
        self.logger.setLevel(self.LEVEL_MAP[current_level.upper()])
        self.logger.propagate = False  # 防止传播到根日志记录器

        self.file_handler, file_path = self.configure_handlers(
            to_file,
            folder, run_name
        )

        # 只在主进程记录日志创建信息
        if self.is_master:
            self.logger.info(
                f"Logger {self.name} created at: {os.path.abspath(file_path if file_path else 'console only')}"
                )
            self.logger.info(f"Master log level: {level}, Worker log level: {rank_level}")

    def configure_handlers(self, to_file, folder, run_name):
        if not self.logger.handlers:
            # 控制台处理器 - 所有进程都使用
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(LoggerFormatter(datefmt = "%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(console_handler)

        file_handler, file_path = None, None
        if to_file:
            file_handler, file_path = self._add_file_handler(folder, run_name)

        return file_handler, file_path

    def _add_file_handler(self, folder, run_name) -> tuple:
        """
        为所有进程添加文件处理器，使用主进程创建的文件名
        """
        if self.is_master:
            os.makedirs(folder, exist_ok = True)
            file_path = os.path.join(
                folder,
                f"{run_name + '_' if run_name else ''}{now()}.log"
            )
            open(file_path, 'a').close()
        else:
            file_path = None

        if torch.distributed.is_initialized():
            file_path_list = [file_path]
            torch.distributed.broadcast_object_list(file_path_list, src = 0)
            file_path = file_path_list[0]

        # 配置文件处理器 - 所有进程使用相同的文件路径
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(LoggerFormatter(datefmt = "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(file_handler)

        return file_handler, file_path

    def log(
            self,
            text: str,
            level: str = "INFO",
    ) -> None:
        log_level = self.LEVEL_MAP.get(level.upper(), logging.INFO)
        if self.logger.isEnabledFor(log_level):
            self.logger.log(log_level, text, stacklevel = 3)

    def debug(self, obj: any) -> None:
        self.log(str(obj), "DEBUG")

    def info(self, obj: any) -> None:
        self.log(str(obj), "INFO")

    def warning(self, obj: any) -> None:
        self.log(str(obj), "WARNING")

    def error(self, obj: any) -> None:
        self.log(str(obj), "ERROR")

    def critical(self, obj: any) -> None:
        self.log(str(obj), "CRITICAL")

    @master_only
    def just_print(self, obj: any, end: str = "\n", time_stamp = False, to_file: Optional[bool] = None) -> None:
        if time_stamp:
            obj = f"[{now()}] {obj}"
        print(obj, end=end, flush=True)
        if to_file is None:
            to_file = self.to_file
        if to_file and self.file_handler:
            self.file_handler.stream.write(f"{obj}{end}")
            self.file_handler.stream.flush()

    def close_file_handler(self) -> None:
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
