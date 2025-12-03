import argparse
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from wall_e import (BaseModel, Evaluator, DumpResults,
                    BaseMapDataset, EarlyStopCallBack,
                    Runner, load_cfg, registry,
                    BaseMetric, Namespace)


Tokenizer = AutoTokenizer.from_pretrained("./tokenizer/smiles")
generation_config = Namespace(
    max_length = 128,
    num_beams = 5,
    eos_token_id = 13,
    vocab_size = len(Tokenizer),
    early_stopping = True,
)


def pad_sequence_left(sequences, batch_first=True, padding_value=-100):
    reversed_sequences = [seq.flip(0) for seq in sequences]
    padded = pad_sequence(
        reversed_sequences,
        batch_first=batch_first,
        padding_value=padding_value
    )
    return padded.flip(1)


def extract_smiles_from_ids(input_ids, tokenizer):
    """提取 [CLS] ... [SEP] 之间的 SMILES 字符串"""
    cls_id = 12
    sep_id = 13
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    # 查找 [CLS] 和 [SEP]
    try:
        start = input_ids.index(cls_id) + 1
    except ValueError:
        start = 0
    try:
        end = input_ids.index(sep_id)
    except ValueError:
        end = len(input_ids)

    smiles_ids = input_ids[start:end]
    smiles = tokenizer.decode(smiles_ids, skip_special_tokens=True)
    return smiles.strip()


def collate_fn(data):
    ids = [item['input_ids'] for item in data]
    ids = pad_sequence_left(
        ids,
        batch_first=True,
        padding_value=Tokenizer.pad_token_id,
    )
    input_ids = ids[:, :-1]
    labels = ids[:, 1:]
    pad_mask = input_ids != Tokenizer.pad_token_id

    return {
        'input_ids': input_ids,
        'attention_mask': pad_mask, 
        'labels': labels
    }


def collect_test_fn(data):
    # list 每一个item是{'text', 'input_id'}
    ids = [item['input_ids'] for item in data]
    ids = [seq[:16] for seq in ids]
    input_ids = pad_sequence_left(
        ids,
        batch_first = True,
        padding_value = Tokenizer.pad_token_id,
    )
    pad_mask = torch.ne(input_ids, Tokenizer.pad_token_id)

    return {
        'input_ids': input_ids,
        'attention_mask': pad_mask,
    }


from transformers import GPT2LMHeadModel, GPT2Config

class GPT(BaseModel):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = cfg.model.max_len

        # === 使用 transformers 自带 GPT2 ===
        gpt_config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=cfg.model.max_len,
            n_embd=cfg.model.d,
            n_layer=cfg.model.num_layers,
            n_head=cfg.model.n,
            bos_token_id=tokenizer.cls_token_id if tokenizer.cls_token_id else 12,
            eos_token_id=tokenizer.sep_token_id if tokenizer.sep_token_id else 13,
            pad_token_id=tokenizer.pad_token_id
        )

        self.model = GPT2LMHeadModel(gpt_config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        logits = outputs.logits
        return dict(logits_ids=logits)

    def train_step(self, data_batch):
        input_ids = data_batch["input_ids"]
        attention_mask = data_batch["attention_mask"]
        labels = data_batch["labels"]

        outputs = self.forward(input_ids, attention_mask, labels)
        loss_output = self.compute_loss(outputs["logits_ids"], labels)
        
        return outputs | loss_output


    def valid_step(self, *args, **kwargs):
        return self.train_step(*args, **kwargs)

    def test_step(self, data_batch, **kwargs):
        input_ids = data_batch["input_ids"]
        attention_mask = data_batch["attention_mask"]

        # === GPT-2 自带 generate 方法 ===
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=generation_config.max_length,
            num_beams=generation_config.num_beams,
            early_stopping=generation_config.early_stopping,
            eos_token_id=generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return {"generated_ids": generated}

    def compute_loss(self, logits_ids, labels) -> dict:
        loss = self.criterion(logits_ids.view(-1, logits_ids.size(-1)), labels.view(-1))
        return dict(loss = loss)



# 验证集的metric
class AccuracyMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.results = []

    def process(self, data_batch, data_samples) -> None:
        # data_samples = {"logit": ..., "labels": ...}
        logits_ids = data_samples['logits_ids']
        labels = data_batch['labels']
        pred = logits_ids.argmax(dim=-1)
        mask = labels != Tokenizer.pad_token_id
        acc = ((pred == labels) & mask).sum().item() / mask.sum().item()
        self.results.append(acc)

    def compute_metrics(self, results) -> dict:
        # 平均所有 batch 的准确率
        accuracy = sum(results) / len(results) if results else 0.0
        return {'accuracy': accuracy}


class DumpValidResult(DumpResults):
    def process(self, data_batch, predictions) -> None:
        # 有需要可以把这两也转为smiles进行分析
        self.results.append({
            'logits_ids': predictions['logits_ids'],
            'labels': data_batch['labels'],
        })


# ====================
# 测试集的metric
class ValidityMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.dataset_meta = 'unimol custom test dataset'


    def process(self, data_batch, data_samples) -> None:
        pred = data_samples["generated_ids"]
        # 提取并解码每个样本
        pred_smiles = []
        for ids in pred:
            smiles = extract_smiles_from_ids(ids, Tokenizer)
            pred_smiles.append(smiles)
        self.results.append({"pred_smiles": pred_smiles})

    def compute_metrics(self, results) -> dict:
            valid = []
            smiles = []
            from rdkit import Chem
            from rdkit import RDLogger
    
            # --- Optional: Suppress RDKit console errors ---
            # This will stop the "SMILES Parse Error" messages from flooding your console
            logger = RDLogger.logger()
            # logger.setLevel(RDLogger.CRITICAL)
            # ----------------------------------------------
    
            # 遍历 batch
            for result in results:
                pred_smiles = result["pred_smiles"]
                for smi in pred_smiles:
                    # Check the return value directly
                    mol = Chem.MolFromSmiles(smi)
                    
                    if mol is not None:
                        # Successfully parsed (valid)
                        valid.append(1)
                        # print("11111") # Your debug print
                        smiles.append((smi, True))
                    else:
                        # Failed to parse (invalid)
                        valid.append(0)
                        # print("00000") # Your debug print
                        smiles.append((smi, False))
    
            # --- Optional: Restore logger state ---
            logger.setLevel(RDLogger.INFO)
            # --------------------------------------
    
            validity = sum(valid) / len(valid) if valid else 0.0
            registry.register('pred_smiles', smiles)
            return dict(validity=validity)


class NoveltyMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.train_smiles = registry.get('train_smiles')

    def process(self, data_batch, data_samples):
        pass

    def compute_metrics(self, results: list) -> dict:
        """统计新颖性 (Novelty)，只计算合法分子"""
        # [(smi, is_valid), ...]
        pred_smiles_all = registry.get('pred_smiles')
        valid_smiles = [smi for smi, is_valid in pred_smiles_all if is_valid]

        if not valid_smiles:
            return {'novelty': 0.0}

        novel = [smi for smi in valid_smiles if smi not in self.train_smiles]
        novelty = len(novel) / len(valid_smiles)
        return {'novelty': novelty}


class UniquenessMetric(BaseMetric):
    def process(self, data_batch, data_samples):
        pass

    def compute_metrics(self, results: list) -> dict:
        """统计唯一性 (Uniqueness)，只计算合法分子"""
        # [(smi, is_valid), ...]
        pred_smiles_all = registry.get('pred_smiles')
        valid_smiles = [smi for smi, is_valid in pred_smiles_all if is_valid]

        if not valid_smiles:
            return {'uniqueness': 0.0}

        unique_smiles = set(valid_smiles)
        uniqueness = len(unique_smiles) / len(valid_smiles)
        return {'uniqueness': uniqueness}


class DumpTestResult(DumpResults):
    def process(self, data_batch, data_samples):
        """保存测试集生成的全部分子和输入 SMILES"""
        input_ids = data_batch['input_ids']  
        pred_smiles_all = registry.get('pred_smiles')

        # 将输入 input_ids 转为 SMILES
        input_smiles = []
        for ids in input_ids:
            smi = extract_smiles_from_ids(ids, Tokenizer)
            input_smiles.append(smi)

        self.results.append({
            'input_smiles': input_smiles,
            'generated_smiles': pred_smiles_all
        })


if __name__ == '__main__':
    # map返回值要是dict
    # filter保留的是判断为True的
    # dataset的数据都是字典，里面的是feature，len(data)是错的，要先抽出来
    # 数据的类型，里面包含什么先看
    # hf map filter的cache数据范围不同不报存，注意basemapdataset会自动shuffle，所以数据范围都不一样
    # 生成时给的attention_mask是(bs, len) ,其合理性在于广播时

    cfg = load_cfg('./gpt.yaml')

    dataset = BaseMapDataset(
        data_source = "./dataset/build/smiles"
    )

    train_smiles = set(dataset.get_split('test')["smiles"])
    registry.register('train_smiles', train_smiles)

    dataloader_args = {
        'batch_size': cfg.training.batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 8,
    }
    train_dataloader = dataset.get_subset("train", n = 8192000).get_batch_loader(
        **dataloader_args, collate_fn=collate_fn, shuffle = True
    )
    valid_dataloader = dataset.get_subset("valid", n = 1024).get_batch_loader(
        **dataloader_args, collate_fn=collate_fn, shuffle = True
    )
    test_dataloader = dataset.get_subset("test", n=1024).get_batch_loader(
        **dataloader_args, shuffle = False,
        collate_fn=collect_test_fn
    )

    causal_mask = torch.tril(
        torch.ones((1, cfg.model.max_len, cfg.model.max_len), dtype = torch.bool)
    )

    model = GPT(cfg, Tokenizer)
    valid_evaluator = Evaluator([
        AccuracyMetric(),
        DumpValidResult(output_dir='result/valid')
    ])
    test_evaluator = Evaluator([
        ValidityMetric(),
        UniquenessMetric(),
        NoveltyMetric(),
        DumpTestResult(output_dir='result/test')
    ])

    runner = Runner(
        train_data_loader = train_dataloader,
        valid_data_loader = valid_dataloader,
        test_data_loader=test_dataloader,
        model = model,
        epochs = cfg.training.epochs,
        valid_evaluator = valid_evaluator,
        test_evaluator=test_evaluator,
        cfg = cfg,
    )
    # callbacks = [
        # EarlyStopCallBack(runner = runner, monitor = "accuracy", greater_is_better = True,
                          # patience = 2000, delta=0.001, mode="batch")
    # ]
    # runner.extend_callbacks(callbacks)

    runner.fit()
    # runner.test()
