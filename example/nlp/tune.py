"""

"""
from typing import Any, Sequence

import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from ray import train

from wall_e import *
import torch.distributed as dist

from model.transformer import TransformerDecoder


class CustomDataset(Dataset):
    def __init__(self, vocab_n, max_len, data_n):
        self.vocab_n = vocab_n
        self.max_len = max_len
        self.data_n = data_n

    def __len__(self):
        return self.data_n

    def __getitem__(self, idx):
        n = torch.randint(low = 1, high = self.vocab_n - self.max_len, size = (1,)).item()
        x = torch.arange(n, n + self.max_len)

        return x


def collate_fn(batch):
    def make_mask(input_ids, pad):
        def look_ahead_mask(size):
            attn_shape = (1, size, size)
            triu_mask = torch.triu(torch.ones(attn_shape), diagonal = 1).to(torch.uint8)
            return triu_mask

        pad_mask = (input_ids == pad).unsqueeze(-2)
        look_ahead_mask = look_ahead_mask(input_ids.shape[-1])
        mask = pad_mask | look_ahead_mask
        return mask

    data = torch.stack(batch)
    input_ids = data[:, :-1]
    label = data[:, 1:]

    pad = 0
    mask = make_mask(input_ids, pad)

    return {
        'input_ids': input_ids,
        'mask': mask,
        'label': label
    }


class TestDataset(Dataset):
    def __init__(self, vocab_n, max_len, data_n, seed_len = 10):
        """

        Args:
            vocab_n:
            max_len: 最大生成长度
            data_n: 测试样本批次数
            seed_len: 起始序列长度
        """
        self.data_n = data_n

        # 预生成可复现测试样本
        self.samples = []
        for idx in range(data_n):
            torch.manual_seed(idx)
            n = torch.randint(low = 1, high = vocab_n - max_len, size = (1,)).item()

            full_seq = torch.arange(n, n + max_len)
            seed_seq = full_seq[:seed_len]

            self.samples.append((seed_seq, full_seq))

    def __len__(self):
        return self.data_n

    def __getitem__(self, idx):
        """
        返回:
            seed_seq: 模型输入的起始序列 (长度=seed_len)
            full_seq: 完整目标序列 (长度=max_len) 用于验证
        """
        return self.samples[idx]


def collate_test_fn(batch):
    seed, full = zip(*batch)
    seed = torch.stack(seed)
    full = torch.stack(full)
    return {
        "input_ids": seed,
        "label": full,
    }


from torch import nn


class DemoNet(BaseModel):
    def __init__(self, transformer, logit_generator):
        super().__init__()
        self.transformer = transformer
        self.logit_generator = logit_generator
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, label, mask = None, *args, **kwargs):
        out_prob, past_key_values = self.transformer(input_ids, mask = mask)
        logit = self.logit_generator(out_prob)
        loss = self.compute_loss(logit, label) if label else None
        return dict(
            logit = logit,
            loss = loss
        )

    def compute_loss(self, logit, label):
        loss = self.criterion(logit.view(-1, logit.size(-1)), label.view(-1))
        return loss

    def train_step(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def valid_step(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def test_step(self, input_ids, **kargs):
        return self.transformer.generate(
            input_ids,
            generator = self.logit_generator,
            max_length = cfg.data.max_len
        )


class ValidMetric(BaseMetric):
    def __init__(self, collect_device = 'cpu', collect_dir = './tmp'):
        super(ValidMetric, self).__init__(collect_device, collect_dir = collect_dir)
        self.dataset_meta = 'my_dataset'
        self.prefix = 'valid'

    def process(self, data_batch, data_samples) -> None:
        pred = data_samples["logit"].argmax(dim = -1)
        label = data_batch['label']
        self.results.append([pred, label])

    def compute_metrics(self, results) -> dict:
        acc = []
        # 这里的result是每一个batch的结果
        for result in results:
            pred = result[0]
            label = result[1]
            acc.append((pred == label).float().mean().cpu())

        acc = (sum(acc) / len(acc)).item()

        return dict(acc = acc)


class TestMetric(ValidMetric):
    def __init__(self):
        super().__init__()
        self.prefix = 'test'

    def process(self, data_batch, data_samples) -> None:
        pred = data_samples["pred_ids"]
        label = data_batch['label']
        self.results.append([pred, label])


class DumpValidResult(DumpResults):
    def process(self, data_batch, predictions) -> None:
        logit = predictions['logit'].cpu().numpy()
        label = data_batch['label'].cpu().numpy()
        pred_ids = logit.argmax(axis = -1)
        self.results.append(
            {
                'logit': logit,
                'label': label,
                'pred_ids': pred_ids
            }
        )


class DumpTestResult(DumpResults):
    def process(self, data_batch: Any, predictions: dict) -> None:
        pred_ids = predictions["pred_ids"].cpu().numpy()
        label = data_batch['label'].cpu().numpy()
        self.results.append(
            {
                'pred_ids': pred_ids,
                'label': label
            }
        )




if __name__ == '__main__':
    from wall_e.tunner import update_cfg_by_tune_params, Tuner
    import argparse

    arg = argparse.ArgumentParser()
    arg.add_argument('--cfg', type = str, default = './cfg.yaml')
    args, _ = arg.parse_known_args()
    cfg_path = args.cfg
    cfg = load_cfg(cfg_path)


    def run(tune_cfg):
        global cfg
        cfg = OmegaConf.merge(cfg, tune_cfg)
        print(cfg)

        bench_dataset = CustomDataset(
            vocab_n = cfg.data.vocab_n,
            max_len = cfg.data.max_len,
            data_n = cfg.data.data_n
        )
        bench_loader = DataLoader(
            bench_dataset,
            batch_size = cfg.data.batch_size,
            shuffle = True,
            collate_fn = collate_fn
        )
        test_dataset = TestDataset(
            vocab_n = cfg.data.vocab_n,
            max_len = cfg.data.max_len,
            data_n = cfg.data.data_n,
            seed_len = cfg.data.seed_len
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size = cfg.data.batch_size,
            shuffle = False,
            collate_fn = collate_test_fn
        )
        # FIXME: why顶层报错 找不到model？
        from model.transformer import TransformerForCausalLLM
        model_cfg = cfg.model
        model = registry.get_model_class(cfg.model.type)(**model_cfg)
        logit_generator = nn.Linear(model_cfg.d, model_cfg.vocab_n)
        net = DemoNet(model, logit_generator)
        # net.load_checkpoint('example/transformer_to_copy_str.pth')

        valid_evaluator = Evaluator([ValidMetric(), DumpValidResult(f'./assert/valid.pkl')])
        test_evaluator = Evaluator([TestMetric(), DumpTestResult(f'./assert/test.pkl')])

        runner = Runner(
            train_data_loader = bench_loader,
            valid_data_loader = bench_loader,
            test_data_loader = test_loader,
            model = net,
            epochs = cfg.training.epochs,
            # optimizer = optimizer,
            valid_evaluator = valid_evaluator,
            test_evaluator = test_evaluator,
            cfg = cfg,
        )
        callbacks = [EarlyStopCallBack(runner = runner, monitor = "loss")]
        runner.extend_callbacks(callbacks)

        runner.fit()
        runner.test()
        runner.cleanup_resources()


    tune_cfg = OmegaConf.load('./tune.yaml')
    tunner = Tuner(
        train_func = run,
        tune_cfg = tune_cfg
    )
    tunner.tune()
    ############# test #############
    # 1. online: model load_model_checkpoint-> runner.test()
    # runner.test()
    # 2. offline: 现在的数据变换逻辑还是过于复杂了，容易出错
    # data_samples = load('example/test_result_epoch_4.pkl')
    # data_list = []
    # for batch in bench_loader:
    #     data_list.append(batch)
    # evaluator_offline = Evaluator([MyMetric2()])
    # #
    # print(evaluator_offline.offline_evaluate(data_samples, data_list, chunk_size = 10))
