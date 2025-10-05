import warnings
from functools import wraps

from packaging.version import parse

import warnings

_warning_cache = set()  # 全局缓存已触发警告的消息


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


class Namespace:
    """
    Examples:
        > args_dict = {'name': 'John', 'age': 25, 'city': 'Example City'}
        > namespace_obj = Namespace(args_dict)
    """

    def __init__(self, args: dict):
        for k, v in args.items():
            setattr(self, k, v)

def first_call_warning(key, message: str):
    """生成一个函数，该函数在首次调用时触发警告，后续调用无操作"""
    if key not in _warning_cache:
        warnings.warn(message)
        _warning_cache.add(key)


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
