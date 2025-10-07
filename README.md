# WALL_E

<img src="https://image.tmdb.org/t/p/original/nYs4ZwnJBK4AgljhvzwNz7fpr3E.jpg" width="500"/>

---
**项目文档**：https://walle.zengzhicun.info

## 简介

**WALL_E** 是一个基于 **PyTorch**、**Datasets** 与 **OmegaConf** 的轻量级深度学习框架，强调模块解耦与易扩展。你可以单独使用 `BaseModel`、`BaseDataset`、`load_cfg` 完成最小训练闭环，也可以与 `Runner/Loops`、分布式、回调与评估系统组合，完成完整训练编排。

在典型科研开发中，它帮助你：

- 降低分布式与 DeepSpeed 集成复杂度（自动包装与采样器适配）
- 标准化训练生命周期（回调与日志、检查点、评估、进度）
- 复用常见训练技巧（混合精度、梯度累积、激活检查点、调度器）
- 用 YAML 集中管理超参，实现稳定复现

---

## 核心特性
- **🧱 模块化、解耦设计**：模型、数据、评估、回调、分布式皆可独立演进
- **🧠 训练编排与生命周期**：`Runner` 统一调度 `Train/Valid/Test Loop` 与回调
- **🚀 分布式与 DeepSpeed**：单机/多卡/CPU、DDP 与 DeepSpeed 启动策略
- **📈 评估与指标**：`Evaluator` + 指标注册，支持结果转储
- **⚡ 训练技巧**：FP16、梯度累积、激活检查点、梯度裁剪、学习率调度
- **🧾 日志与追踪**：控制台/文件日志、W&B（可选）、轮次汇总
- **🧰 YAML 配置**：OmegaConf 加载与合并，命令行可覆盖

---

## 安装与环境
- Python 3.9+
- PyTorch 2.3.0+
- 可选：DeepSpeed、WandB、Ray

安装：
```bash
git clone https://github.com/alan-tsang/wall_e.git
cd wall_e
pip install .
```

---

## 快速开始（两种路径）

### 1) 代码最短路径（约 10 分钟）

定义最小模型与数据集，并运行 `Runner`：

```python
from wall_e.model.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyModel(BaseModel):
    def __init__(self, dim=8):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
    def compute_loss(self, x, y):
        pred = self.linear(x)
        return {"loss": F.mse_loss(pred, y)}
    def train_step(self, batch):
        return self.compute_loss(**batch)
    def valid_step(self, batch):
        return self.compute_loss(**batch)
    def test_step(self, batch):
        return self.compute_loss(**batch)
```

构建数据与 DataLoader：

```python
from wall_e.dataset.dataset import BaseMapDataset
from torch.utils.data import DataLoader, TensorDataset
import torch

def make_tensor_ds(n=64, dim=8):
    x = torch.randn(n, dim)
    y = torch.randn(n, 1)
    return TensorDataset(x, y)

class TensorMapDataset(BaseMapDataset):
    def _set_dataset(self, data_source, only_local=False):
        ds = make_tensor_ds()
        return ds

ds = TensorMapDataset(data_source="dummy", split_ratios=(0.9, 0.1))
train_loader = DataLoader(ds.get_split("train"), batch_size=8)
valid_loader = DataLoader(ds.get_split("test"), batch_size=8)
```

运行 `Runner`：

```python
from wall_e.runner.runner import Runner
from omegaconf import OmegaConf

cfg = OmegaConf.create({
  "run_name": "quickstart",
  "training": {
    "fp16": False,
    "progress_every_n_batches": 10,
  },
  "optimizer": {"lr": 1e-3, "weight_decay": 0.0}
})

runner = Runner(
  model=ToyModel(),
  epochs=2,
  train_data_loader=train_loader,
  valid_data_loader=valid_loader,
  cfg=cfg,
)
runner.fit()
```

预期：控制台将打印进度与轮次汇总，并在 `logs/` 与 `checkpoints/` 生成输出。

### 2) 基于 YAML 的配置化启动

新建 `demo.yaml`：

```yaml
# demo.yaml
dataset:
  type: 'YourMapDataset'   # 由你实现的数据集，或注册后复用
  params:
    data_source: './dataset/your_path'
    shuffle: true
    split_ratios: [0.98, 0.01, 0.01]

model:
  type: 'YourModel'        # 由你实现或复用的模型
  params:
    vocab_size: 32000
    hidden_size: 512

run_name: 'quickstart-demo'
run_description: "Minimal runnable example"

training:
  epochs: 1
  gradient_accumulation: 1
  activation_checkpoint: []
  grad_clip: null
  fp16: false
  valid_begin_epoch: 1
  valid_interval_epoch: 1
  test_begin_epoch: 1
  test_interval_epoch: 1
  progress_every_n_epochs: 1
  progress_every_n_batches: 1

log:
  to_file: true
  folder: "./assert/logs"
  level: "INFO"
  rank_level: "WARNING"

pt:
  enable: true
  dir: "./assert/checkpoints"
  best_monitor: { loss: true }
  topk: 3
  begin_epoch: 1
  epoch_interval: 1

wandb:
  enable: false
  proj_name: "wall_e quickstart"
  offline: true
  dir: "./assert"
  tags: ["wall_e", "quickstart"]
```

编写启动脚本 `run_demo.py`：

```python
from wall_e.config.load_config import load_cfg
from wall_e.runner.runner import Runner
from wall_e.model.base_model import BaseModel
from wall_e.dataset.dataset import BaseMapDataset
from torch.utils.data import DataLoader

cfg = load_cfg('demo.yaml')

# 按需构建 dataset / model（或通过注册表从 cfg 构建）
ds = BaseMapDataset.from_cfg(cfg.dataset.path, metadata=getattr(cfg.dataset, 'metadata', None))
train_loader = DataLoader(ds.get_split('train'), batch_size=8)
valid_loader = DataLoader(ds.get_split('test'), batch_size=8)

model = BaseModel.from_cfg(cfg.model.params | {'type': cfg.model.type})

runner = Runner(model=model, epochs=cfg.training.epochs,
                train_data_loader=train_loader, valid_data_loader=valid_loader, cfg=cfg)
runner.fit()
```

运行：
```bash
python run_demo.py
# 或分布式（单机多卡）：
torchrun --nproc_per_node=<NUM_GPUS> run_demo.py
```

---

## 模块与 API（速览）
- **Runner**（`wall_e.runner.runner.Runner`）：训练执行引擎，装配回调、评估、循环与优化器。
- **Loops**（`wall_e.runner.loop.*`）：`TrainLoop`、`ValidLoop`、`TestLoop`，按 epoch/iter 调度；可自定义替换。
- **Dataset**（`wall_e.dataset.*`）：`BaseMapDataset`、`BaseIterableDataset`，统一批加载与流水线。
- **Evaluator & Metric**（`wall_e.eval.*`）：指标注册与汇报，支持结果转储。
- **Config**（`wall_e.config.*`）：OmegaConf 加载与合并，集中化管理参数。
- **Logging**（`wall_e.logging.*`）：控制台/文件日志，主从 rank 区分级别。
- **Distributed**（`wall_e.dist.*`）：设备与策略管理，DDP/DeepSpeed 包装与采样器注入。

更多详见文档：`教程 → 概述/快速开始` 与 `API → 概览/Runner/Dataset/Loops`。

---

## 示例与扩展
- 参考 `example/` 与 `wall_e_doc/docs/tutorials` 获取更多脚本与配置示例。
- 通过继承 `BaseModel`/`BaseDataset`、实现自定义回调或指标扩展能力；并使用注册表以配置化方式启用。

---

## 贡献与致谢
- 欢迎提交 **PR** 或 **issue**，参与代码与文档完善。
- 许可证详见 `LICENSE`。
- 致谢 **PyTorch**、**DeepSpeed**、**WandB**、**Ray**、**HuggingFace Datasets** 等社区。
