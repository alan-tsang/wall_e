import pytest
from unittest.mock import patch, MagicMock
import os
from omegaconf import OmegaConf
from wall_e.callback.checkpoint_callback import CheckpointCallback


# 定义模拟的 Runner 类
class MockRunner:
    def __init__(self):
        self.cfg = OmegaConf.create(
            {
                "pt": {
                    "begin_epoch": 1,
                    "epoch_interval": 1,
                    "begin_batch": 1,
                    "batch_interval": 1,
                    "dir": "./checkpoints",
                    "best_monitor": {"loss": False},
                    "topk": 3
                },
                "run_name": "test_run"
            }
        )
        self.is_main_process = True
        self.is_deepspeed = False
        self.logger = MagicMock()
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.scheduler = MagicMock()
        self.state = MagicMock()
        self.state.current_epoch = 1
        self.state.current_batch = 1
        self.state.current_step = 1


# 测试保存标准检查点的功能
@patch('torch.save')
@patch('torch.get_rng_state')
@patch('os.makedirs')
def test_save_standard_checkpoint(mock_makedirs, mock_get_rng_state, mock_save):
    runner = MockRunner()
    callback = CheckpointCallback(runner)
    name = "test_checkpoint"
    save_path = callback._save_standard_checkpoint(name)

    # 验证保存路径是否正确
    expected_path = os.path.join(callback.folder, f"{name}.pt")
    assert save_path == expected_path
    # 验证 torch.save 是否被调用一次
    assert mock_save.call_count == 1


# 测试训练开始前的回调
@patch('torch.save')
@patch('torch.get_rng_state')
@patch('os.makedirs')
def test_before_train(mock_makedirs, mock_get_rng_state, mock_save):
    runner = MockRunner()
    callback = CheckpointCallback(runner)
    callback.before_train()

    # 验证是否保存了初始模型
    assert mock_save.call_count == 1
    # 验证日志信息
    runner.logger.info.assert_called_with("初始模型保存成功: initial")


# 测试训练结束后的回调
@patch('torch.save')
@patch('torch.get_rng_state')
@patch('os.makedirs')
def test_after_train(mock_makedirs, mock_get_rng_state, mock_save):
    runner = MockRunner()
    callback = CheckpointCallback(runner)
    callback.after_train()

    # 验证是否保存了最终模型
    assert mock_save.call_count == 1
    # 验证日志信息
    runner.logger.info.assert_called_with("最终模型保存成功: final")


# 测试每个 epoch 运行后的回调
@patch('torch.save')
@patch('torch.get_rng_state')
@patch('os.makedirs')
def test_after_running_epoch(mock_makedirs, mock_get_rng_state, mock_save):
    runner = MockRunner()
    callback = CheckpointCallback(runner)
    callback.after_running_epoch()

    # 验证保存了两个检查点（epoch 和 latest）
    assert mock_save.call_count == 2
    # 验证 epoch 检查点的日志信息
    expected_epoch_path = os.path.join(callback.folder, f"epoch_{runner.state.current_epoch}.pt")
    runner.logger.info.assert_any_call(f"检查点保存至: {expected_epoch_path}")
    # 验证 latest 检查点的日志信息
    expected_latest_path = os.path.join(callback.folder, "latest.pt")
    runner.logger.info.assert_any_call(f"检查点保存至: {expected_latest_path}")