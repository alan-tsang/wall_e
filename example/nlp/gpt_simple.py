from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from wall_e import *
from wall_e.model.transformer import (TransformerForCausalLM, GenerationConfig,
                                        get_generation_strategy)

generation_config = GenerationConfig(
    method = 'greedy',
    max_length = 36,
    eos_token_id = 1,
    early_stopping = True,
)
generation_method = get_generation_strategy(generation_config)

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
        attention_mask = pad_mask | look_ahead_mask
        
        attention_mask = 1 - attention_mask

        return attention_mask

    data = torch.stack(batch)
    input_ids = data[:, :-1]
    label = data[:, 1:]

    pad = 0
    attention_mask = make_mask(input_ids, pad)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
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
    pad = 0
    attention_mask = (seed != pad).to(torch.uint8)
    return {
        "input_ids": seed,
        "attention_mask": attention_mask,
        "label": full,
    }



from torch import nn


class DemoNet(BaseModel):
    def __init__(self, transformer, logit_generator):
        super().__init__()
        self.transformer = transformer
        self.logit_generator = logit_generator
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask = None, *args, **kwargs):
        outputs = self.transformer(input_ids, attention_mask = attention_mask)
        logits = outputs['hidden_states']
        past_key_values = outputs['past_key_values']
        logit = self.logit_generator(logits)

        return dict(logits_ids = logit, past_key_values = past_key_values)

    def train_step(self, data_batch):
        input_ids = data_batch["input_ids"]
        attention_mask = data_batch["attention_mask"]
        label = data_batch["label"]
        outputs = self.forward(input_ids, attention_mask)
        loss_output = self.compute_loss(outputs["logits_ids"], label)
        return outputs | loss_output

    def valid_step(self, *args, **kwargs):
        return self.train_step(*args, **kwargs)

    def test_step(self, data_batch, **kargs):
        input_ids = data_batch["input_ids"]
        attention_mask = data_batch["attention_mask"]

        return generation_method.generate(
            model=self,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def compute_loss(self, logit, label) -> dict:
        loss = self.criterion(logit.reshape(-1, logit.shape[-1]), label.reshape(-1))
        return dict(loss = loss)


class ValidMetric(BaseMetric):
    def __init__(self, collect_device = 'cpu', collect_dir = './tmp'):
        super(ValidMetric, self).__init__(collect_device, collect_dir = collect_dir)
        self.dataset_meta = 'my_dataset'
        self.prefix = 'valid'
        self.monitor_metrics = {'acc': True}

    def process(self, data_batch, data_samples) -> None:
        pred = data_samples["logits_ids"].argmax(dim = -1)
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
        pred = data_samples["generated_ids"]
        label = data_batch['label']
        self.results.append([pred, label])


class DumpValidResult(DumpResults):
    def process(self, data_batch, predictions) -> None:
        logit = predictions['logits_ids'].cpu().numpy()
        label = data_batch['label'].cpu().numpy()
        generated_ids = logit.argmax(axis = -1)
        self.results.append(
            {
                'logits_ids': logit,
                'label': label,
                'generated_ids': generated_ids
            }
        )


class DumpTestResult(DumpResults):
    def process(self, data_batch: Any, predictions: dict) -> None:
        generated_ids = predictions["generated_ids"].cpu().numpy()
        label = data_batch['label'].cpu().numpy()
        self.results.append(
            {
                'generated_ids': generated_ids,
                'label': label
            }
        )


if __name__ == '__main__':
    import argparse

    arg = argparse.ArgumentParser()
    arg.add_argument('--cfg', default = 'config/config.yaml')
    args, _ = arg.parse_known_args()
    cfg_path = args.cfg
    cfg = load_cfg(cfg_path)

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

    model_cfg = cfg.model
    model = registry.get_model_class(cfg.model.type)(**model_cfg)
    logit_generator = nn.Linear(model_cfg.d, model_cfg.vocab_n)
    net = DemoNet(model, logit_generator)
    from wall_e.util.dl_util import get_model_info
    # net.load_checkpoint('example/transformer_to_copy_str.pth')

    valid_evaluator = Evaluator([
        ValidMetric(),
        # DumpValidResult(output_dir = f'result/valid')
    ])
    test_evaluator = Evaluator([
        TestMetric(),
        # DumpTestResult(output_dir = f'result/test')
    ])

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

    callbacks = [EarlyStopCallBack(runner = runner, monitor = "loss",
                                   delta = 0.01, patience = 2, greater_is_better = False,
                                   mode = "epoch")]
    runner.extend_callbacks(callbacks)

    runner.fit()
    runner.test()
    runner.cleanup_resources()
