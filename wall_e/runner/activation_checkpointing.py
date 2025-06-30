from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing
)
from operator import attrgetter
# Copyright (c) OpenMMLab. All rights reserved.
from functools import wraps
from operator import attrgetter
from typing import List, Union

import torch
from torch.utils.checkpoint import checkpoint



def turn_on_activation_checkpointing_by_cls(model, modules):
    """
    通过模块类型进行激活值重算的包装
    """
    # 首先获取目标模块的类型集合
    module_types = set()
    for module_path in modules:
        # 从模型中获取指定路径的模块
        module = attrgetter(module_path)(model)
        # 收集模块的实际类型
        module_types.add(type(module))

    # 创建检查函数：检查模块是否属于目标类型
    def check_fn(submodule):
        return type(submodule) in module_types

    # 应用激活检查点
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn
    )

def turn_on_activation_checkpointing_by_obj(model, modules):
    """
    通过具体的对象进行激活值重算的包装
    """
    target_paths = set(modules)

    # 构建模块路径反向索引 {id(module): full_path}
    module_to_path = {}
    for name, module in model.named_modules():
        module_to_path[id(module)] = name

    # 检查函数：当前模块是否在目标路径中
    def check_fn(submodule):
        module_id = id(submodule)
        if module_id not in module_to_path:
            return False
        full_path = module_to_path[module_id]
        return full_path in target_paths

    # 应用激活检查点
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn
    )


def wrap_forward(forward):
    @wraps(forward)
    def wrapper(*args):
        return checkpoint(forward, *args, use_reentrant=False)

    return wrapper

def turn_on_activation_checkpointing(
        model: torch.nn.Module,
        modules: Union[List[str], str]
    ):
    if isinstance(modules, str):
        modules = [modules]
    for module_name in modules:
        module = attrgetter(module_name)(model)
        module.forward = wrap_forward(module.forward)
