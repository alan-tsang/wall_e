# WALL_E

<img src="https://image.tmdb.org/t/p/original/nYs4ZwnJBK4AgljhvzwNz7fpr3E.jpg" width="500"/>

---

## 简介

在深度学习项目开发中，开发者常常面临如下痛点：

- **代码结构混乱**：模型、数据、训练流程高度耦合，难以维护和扩展。
- **分布式训练复杂**：多机多卡环境下的初始化、同步、日志管理繁琐，容易出错。
- **训练技巧集成难**：如混合精度、梯度累积、激活检查点等技巧手动集成成本高。
- **实验管理低效**：超参数调优、实验追踪、结果复现缺乏统一方案。
- **评估与监控分散**：评估指标、日志、进度、早停、模型保存等功能分散在各处，难以统一管理。

**WALL_E** 针对上述痛点，提供了一个**轻量级、模块化、分布式友好**的深度学习开发框架，适用于科研场景。其目标是让你**专注于创新和业务逻辑**，而不是重复造轮子和调试底层细节。

**适用人群：**
- 需要快速搭建和迭代深度学习实验的科研人员和工程师
- 希望在分布式环境下高效训练和管理模型的团队
- 追求代码结构清晰、易于维护和扩展的开发者

**核心价值：**
- 让新手快速上手，专家高效定制
- 让复杂训练流程一键集成，实验管理自动化
- 让分布式和多进程训练像单机一样简单

---

## 核心特性
- **模块化设计**：各组件解耦，便于扩展和复用。你可以像搭积木一样灵活组合模型、数据、训练、评估等模块。
- **生命周期管理**：支持训练各阶段的回调机制，轻松插拔早停、进度、日志、Wandb等功能。
- **分布式训练**：内置分布式环境初始化与通信，主进程、从进程日志集成输出，支持多种分布式后端。
- **训练技巧**：支持16bit混合精度、激活检查点、梯度累积、梯度裁剪、学习率调度、deepspeed等主流训练技巧。
- **模型分析**：FLOPs与激活计数，便于模型复杂度评估和优化。
- **注册机制**：模型、数据集、指标等均可动态注册，方便扩展和自定义。
- **灵活评估**：自定义评估器与指标体系，支持多任务和多指标评估。
- **训练监控**：日志、进度、早停、检查点、自动上传，训练过程全方位可控。
- **超参调优**：集成ray，支持多实验与自动调参，提升实验效率。
- **YAML配置**：参数集中管理，支持命令行覆盖，实验可复现性强。

---

## 主要组件
- **模型**
  - Transformer（编码器/解码器/LLM）
  - 图神经网络（GIN、GCN、GAT、GraphSAGE、EGNN）
  - MoE（混合专家）
- **数据集**
  - BaseDataset（抽象基类，统一数据接口）
  - MapDataset（适用于可随机访问的数据集）
  - IterableDataset（适用于流式大数据集）
- **评估**
  - BaseMetric（评估指标基类）、Evaluator（评估器）、DumpResults（结果保存）等
- **回调**
  - CheckpointCallback（模型检查点）、EarlyStopCallBack（早停）、WandbCallback（Wandb日志）、ProgressCallBack（进度显示）

---

## 环境依赖
- Python 3.9+
- PyTorch 2.3.0+
- 支持 CPU/GPU
- 可选：Deepspeed、Wandb、Ray

---

## 安装
```bash
git clone https://github.com/alan-tsang/wall_e.git
cd wall_e
pip install .
```

---

## 快速上手

只需几行代码，即可完成数据加载、模型构建、训练与评估：

```python
from wall_e import Runner, load_cfg
from wall_e.dataset import BaseMapDataset
from model.transformer import TransformerForCausalLLM

dataset = BaseMapDataset('path/to/your/data')
train_loader = dataset.get_batch_loader(batch_size=32, num_workers=4)
val_loader = dataset.get_batch_loader(batch_size=32, num_workers=4)
test_loader = dataset.get_batch_loader(batch_size=32, num_workers=4)

cfg = load_cfg('path/to/your/config.yaml')
model = TransformerForCausalLLM(**cfg.model)

runner = Runner(
    model = model,
    train_data_loader = train_loader,
    val_data_loader = val_loader,
    test_data_loader = test_loader,
    cfg = cfg,
)
runner.fit()      # 训练+验证+测试
runner.test()     # 独立测试
```

---

## 配置与用法说明
- **配置文件**：使用YAML集中管理训练参数，支持命令行动态修改，便于实验复现。
- **数据处理**：通过`huggingface dataset`模块加载和处理数据，支持多种格式和分割方式。
- **模型定义**：所有模型组件继承于`BaseModel`，自定义训练、测试和测试流程。
- **Runner**：统一训练、验证、测试入口，支持灵活的回调和钩子。
- **示例**：详见`examples/gpt.py`和`tune.py`。

---

## 贡献与致谢
- 欢迎提交PR或issue，参与代码和文档完善。
- 许可证信息详见 LICENSE 文件。
- 致谢 MMEngine、LAVIS、PyTorch、Deepspeed、Wandb、Ray 等社区。
