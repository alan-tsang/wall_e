import warnings
from functools import wraps

from packaging.version import parse
import warnings

_warning_cache = set()  # 全局缓存已触发警告的消息

import functools
import time
from typing import Optional, Callable, Any


class Namespace(dict):
    """A dict subclass with attribute (dot) access."""

    # 支持嵌套字典自动转 Namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():

            if isinstance(v, dict) and not isinstance(v, Namespace):
                self[k] = Namespace(v)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Namespace' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if isinstance(value, dict) and not isinstance(value, Namespace):
            value = Namespace(value)
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'Namespace' object has no attribute '{name}'")

    def __repr__(self):
        return f"Namespace({super().__repr__()})"


def set_proxy():
    path = '/home/alan/desktop/project/get_windows_ip.sh'
    cmd = f"bash {path}"
    # 运行
    import subprocess
    result = subprocess.check_output(
        cmd.split(), stderr = subprocess.STDOUT,
        timeout = 30, universal_newlines = True
        )
    return result.rstrip()


def timeit(
    func: Optional[Callable] = None, 
    *,
    repeats: int = 1,
    warmup: int = 0,
    prefix: str = "",
    verbose: bool = True,
    return_time: bool = False
) -> Callable:
    """
    函数执行时间测量装饰器
    
    参数:
        repeats: 重复执行次数 (默认: 1)
        warmup: 预热次数 (不计算时间，默认: 0)
        prefix: 输出信息前缀 (默认: "")
        verbose: 是否打印时间信息 (默认: True)
        return_time: 是否返回执行时间 (默认: False)
    
    使用示例:
        1. 基本用法: 
            @timeit
            def my_func():
                ...
                
        2. 带参数:
            @timeit(repeats=5, warmup=1, prefix="[Benchmark]")
            def my_func():
                ...
                
        3. 获取执行时间:
            result, exec_time = my_func(return_time=True)
    """
    if func is None:
        return functools.partial(
            timeit, 
            repeats=repeats,
            warmup=warmup,
            prefix=prefix,
            verbose=verbose,
            return_time=return_time
        )
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # 处理嵌套的 return_time 请求
        return_time_flag = kwargs.pop('return_time', False) or return_time
        
        # 预热运行 (不计算时间)
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # 实际计时运行
        start_time = time.perf_counter()
        for _ in range(repeats):
            result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # 计算执行时间
        total_time = end_time - start_time
        avg_time = total_time / repeats
        
        # 打印结果
        if verbose:
            time_unit = "ms"
            scale = 1000
            
            if avg_time < 0.001:
                time_unit = "μs"
                scale = 1000000
            elif avg_time > 1:
                time_unit = "s"
                scale = 1
            
            msg = (f"{prefix}[Timeit] {func.__name__} - "
                   f"{repeats} runs, {warmup} warmup: "
                   f"avg {avg_time * scale:.4f} {time_unit} "
                   f"| total {total_time * scale:.4f} {time_unit}")
            from ..logging.util import print_log
            print(msg)
        
        # 返回结果
        if return_time_flag:
            return result, (avg_time, total_time)
        return result
    
    return wrapper


def better_dict_4_print(_dict):
    import json
    import omegaconf
    from omegaconf import OmegaConf

    if isinstance(_dict, omegaconf.dictconfig.DictConfig):
        _dict = OmegaConf.to_container(_dict, resolve=True)
    return json.dumps(_dict, indent=2)


def set_proxy():
    pass

def set_seed(seed: int = 0):
    """设置随机数种子"""
    import random
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now():
    from datetime import datetime
    return str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)
