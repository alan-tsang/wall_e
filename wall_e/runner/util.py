import warnings
import torch
from typing import Callable, List, Optional, Union
import numpy as np
from collections.abc import Iterable, Mapping


def move_data_to_device(data, device):
    # 基础张量类型：直接移动设备
    if torch.is_tensor(data):
        return data.to(device, non_blocking = True)

    # 数字类型：转换为0维张量
    elif isinstance(data, (int, float, bool)):
        return torch.tensor(data, device = device)

    # 可迭代集合类型：递归处理元素
    elif isinstance(data, (list, tuple, set)):
        processed = [move_data_to_device(d, device) for d in data]
        return type(data)(processed)

    # 字典类型：递归处理键值
    elif isinstance(data, Mapping):
        return {k: move_data_to_device(v, device) for k, v in data.items()}

    # NumPy数组：转换为张量后移动
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)

    # 支持.to()方法的自定义对象
    elif hasattr(data, 'to'):
        return data.to(device)

    # 其他无法处理的类型
    else:
        warnings.warn(f"Data type {type(data)} not supported, returning original object")
        return data
