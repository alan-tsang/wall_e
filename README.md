# WALL_E
<img src="https://image.tmdb.org/t/p/original/nYs4ZwnJBK4AgljhvzwNz7fpr3E.jpg" width="500"/>

## 介绍
WALL_E 是一个全面的轻量级深度学习框架，旨在简化模型开发、训练、评估和部署流程。它包含了多种模型架构、训练策略、评估指标以及分布式计算支持，并复现了一些基础模型

## 开发框架特性
- 模块化设计：通过将系统分解为独立的组件，提高代码复用性和可维护性。
- 生命周期管理：基于训练过程的不同阶段提供相应的回调机制。
- 分布式训练支持：内置对分布式训练环境的初始化和通信支持。
- 常用训练技巧支持：
  - fp16/16bit
  - activation_checkpoint
  - gradient_accumulation
  - grad_clip
  - warmup & cosine decay
  - deepspeed
- 模型分析工具：提供FLOPs和激活函数的计数器来评估模型复杂度。
- 注册模式：使用注册器来动态注册和获取模型、数据集、评估指标等。
- 评估系统：灵活的评估器和指标系统，支持自定义指标。
- 日志系统：支持多进程日志记录，主进程独占日志输出。
- 训练监控与管理：支持训练过程中的日志记录、进度追踪、早停、模型检查点保存，自动上传至wandb等功能。
- 调参与多实验：支持通过ray进行超参数调优和多实验管理。
- 配置文件驱动：使用YAML配置文件来管理实验参数，支持命令行动态修改和加载。

## 组件
- **模型**
  - Transformer模型：用于语言模型任务，包括编码器和解码器架构。
  - 图神经网络（GNN）：包括GIN、GCN、GAT和GraphSAGE等图卷积层。
  - EGNN：基于坐标的图神经网络。
  - MoE（混合专家模型）：实现前馈网络的多专家架构。

- **数据集**
  - BaseDataset：抽象基类，定义了数据集的基本操作。
  - MapDataset：适用于可随机访问的数据集。
  - IterableDataset：适用于流式数据集。

- **评估**
  - BaseMetric：评估指标的基类。
  - Evaluator：用于执行评估任务。
  - 包括用于保存结果的DumpResults等具体指标类。

- **回调**
  - CheckpointCallback：保存模型检查点。
  - EarlyStopCallBack：早停回调。
  - WandbCallback：支持Weights & Biases日志记录。
  - ProgressCallBack：显示训练进度。

## 环境
- Python
- PyTorch 1.1+
- 支持CPU和GPU训练
- 可选依赖：Deepspeed、Wandb、ray

## 安装
```bash
git clone https://gitee.com/zengton/wall_e.git
cd wall_e
pip install -e .
```

## 使用
```python
from wall_e import Runner, load_cfg
from wall_e.dataset import BaseMapDataset
from model.transformer import TransformerForCausalLLM

dataset = BaseMapDataset('path/to/your/data')
train_loader = dataset.get_batch_loader(batch_size=32, num_workers=4)
val_loader = dataset.get_batch_loader(batch_size=32, num_workers=4)
test_loader = dataset.get_batch_loader(batch_size=32, num_workers=4)

path = 'path/to/your/config.yaml'
cfg = load_cfg(path)
model = TransformerForCausalLLM(**cfg.model)

runner = Runner(
    model = model,
    train_data_loader = train_loader,
    val_data_loader = val_loader,
    test_data_loader = test_loader,
    cfg = cfg,
)
# train, valid and test
runner.fit()
# independent test
runner.test()

```
- 配置文件：项目使用YAML配置文件来设置训练参数。
- 数据处理：通过dataset模块加载和处理数据。
- 模型定义：模型组件在model目录下定义。
- Runner：用于执行训练、验证和测试任务。
- 调用`fit`方法来启动全实验过程。
- 调用`test`方法启动你需要的测试

查看示例目录中`gpt.py`和`tune.py`了解具体的使用方式。

## 许可证
本项目的具体许可证信息请参阅LICENSE文件。

## 贡献
欢迎贡献代码和建议。请提交PR或issue到项目仓库。

## 致谢
本项目受到了许多开源项目的启发和支持，特别是MMEngine、LAVIS、PyTorch、Deepspeed、Wandb、Ray等社区的贡献者们。