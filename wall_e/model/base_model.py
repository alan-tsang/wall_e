"""
adapted from salesforce's lavis: https://github.com/salesforce/LAVIS/blob/main/lavis/models/base_model.py
"""
import inspect
import logging
import re
import warnings
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()


    def load_checkpoint(self, cached_path):
        """Load from a pretrained checkpoint.
        Maybe this should expect no mismatch in the model keys and the checkpoint keys.
        """

        checkpoint = torch.load(cached_path, map_location="cpu")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("load checkpoint from %s" % cached_path)
        warnings.warn("Missing keys {}".format(msg.missing_keys))

        return msg


    @classmethod
    def from_cfg(cls, cfg):

        """根据配置文件动态构建模型，支持以下功能：
        - 自动匹配子类构造器（当继承BaseModel时）
        - 预训练权重加载
        - 设备分配

        example:
        >>> cfg = {
        ... "type": "TransformersToy",
        ... "layers": 6
        ... }
        >>> model = BaseModel.from_cfg(cfg)

        >>> cfg = {
        ... "layers": 6
        ... }
        >>> model = TransformersToy.from_cfg(cfg)

        """

        # 解析配置类型（支持OmegaConf/argparse/dict）
        if hasattr(cfg, 'get'):
            # OmegaConf等类dict对象
            config_dict = dict(cfg)
        else:
            config_dict = vars(cfg) if not isinstance(cfg, dict) else cfg

        # 动态获取子类构造器（允许继承时自动匹配）
        if 'type' in config_dict:
            # 多态 + 注册机制
            """
            model = BaseModel.from_cfg(cfg)
            """
            from ..common.registry import registry
            registry.get_model_class(config_dict['type'])
            model_cls = registry.get_model_class(config_dict['type'])

        elif cls != BaseModel:
            # 如果直接通过子类调用，使用子类自身
            """
            MyTransformerLM.from_cfg(cfg)
            """
            model_cls = cls
        else:
            raise ValueError("Must specify type when using BaseModel directly")

        valid_args = inspect.signature(model_cls.__init__).parameters
        model_args = {k: v for k, v in config_dict.items() if k in valid_args}

        model = model_cls(**model_args)

        if 'pretrained' in config_dict:
            load_result = model.load_checkpoint(config_dict['pretrained'])
            # 检查权重加载（根据需求调整）
            if len(load_result.missing_keys) > 0 and not config_dict.get('allow_missing_keys', False):
                import warnings
                warnings.warn(f"Missing keys in checkpoint: {load_result.missing_keys}")

        # 设备分配（优先使用配置文件指定设备）
        device = config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(device, str):
            device = torch.device(device)
        model = model.to(device)

        return model


    def freeze_parameters(self, freeze = True, regex = None):
        """参数冻结控制
        Args:
            freeze (bool): 冻结/解冻
            regex (str): 参数名正则匹配模式
        Example:
            >>> model.freeze_parameters(regex="^embeddings")
        """

        def _match(name):
            return re.fullmatch(regex, name) if regex else True

        for name, param in self.named_parameters():
            if _match(name):
                param.requires_grad = not freeze
        return self


    def visualize_architecture(self,
                               input,
                               save_path: str = "model_graph.png"):
        """可视化模型计算图（需要安装torchviz）"""
        from torchviz import make_dot

        output = self(**input)
        graph = make_dot(output, params = dict(self.named_parameters()))
        graph.render(save_path, format = 'png', cleanup = True)
        print(f"Graph saved to {save_path}")


    def plot_parameter_histogram(self, param_name: str, bins: int = 50):
        from matplotlib import pyplot as plt
        """绘制参数分布直方图"""
        param = dict(self.named_parameters())[param_name]
        data = param.detach().cpu().numpy().flatten()

        fig, ax = plt.subplots()
        ax.hist(data, bins = bins, alpha = 0.7)
        ax.set_title(f"Parameter Distribution: {param_name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        return fig


    def detect_parameter_outliers(self, sigma = 3) -> List[Dict]:
        """检测参数异常值（基于3σ原则）"""
        outliers = []
        for name, param in self.named_parameters():
            data = param.detach().cpu().numpy().flatten()
            mean, std = np.mean(data), np.std(data)
            threshold = sigma * std
            outlier_indices = np.where(np.abs(data - mean) > threshold)[0]

            if len(outlier_indices) > 0:
                outliers.append(
                    {
                        "parameter": name,
                        "total": len(data),
                        "outliers": len(outlier_indices),
                        "ratio": len(outlier_indices) / len(data),
                        "max_val": np.max(data),
                        "min_val": np.min(data)
                    }
                )
        return outliers


    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

