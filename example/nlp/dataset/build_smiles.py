from datasets import load_dataset
from rdkit import Chem
from wall_e import BaseMapDataset, set_proxy

def standard_smiles(smiles_list):
    ans = []
    for smiles_i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles_i, sanitize=True)
            ans_smiles_i = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            ans.append(ans_smiles_i)
        except Exception:
            print(f"Unstandard SMILES: {smiles_i}")
    return ans

def process_smiles(dataset):
    smiles_list = dataset['smi']
    smiles_list = standard_smiles(smiles_list)
    dataset['smiles'] = smiles_list
    return dataset

def filter_smiles(example_batch):
    smiles_list = example_batch['smiles']
    # 返回一个布尔列表，表示哪些 SMILES 是有效的
    keep_mask = [smi is not None and len(smi.strip()) > 0 for smi in smiles_list]
    return keep_mask


def tokenize(dataset):
    smiles_list = dataset['smiles']
    tokenized = tokenizer(
        smiles_list,
        padding=False,
        truncation=True,
        max_length=512
    )
    dataset['input_ids'] = tokenized['input_ids']
    return dataset

def filter_long_smiles(example_batch, max_length=128):
    input_ids_list = example_batch["input_ids"]
    keep_mask = [len(ids) <= max_length for ids in input_ids_list]
    return keep_mask


if __name__  == '__main__':
    from transformers import AutoTokenizer
    model_checkpoint = './tokenizer/smiles'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    raw_dataset = load_dataset(
        "csv",
        data_files = "./dataset/raw/clean_smi.csv",
        column_names = ["smi"]
    )["train"]
    dataset = BaseMapDataset(
        data_source = raw_dataset,
        shuffle = True,
        split_ratios = (0.99, 0.005, 0.005),
    )
    # 首先处理 SMILES，但不移除列
    dataset = dataset.map(
        process_smiles,
        batch_size = 1024,
        num_proc = 8,
        batched = True,
        remove_columns = ["smi"]
    )
    # 然后过滤无效的 SMILES
    dataset = dataset.filter(
        filter_smiles,
        batched = True,
        batch_size = 1024,
        num_proc = 8
    )
    # 离线SMILES分词
    dataset = dataset.map(
        tokenize,
        batched = True,
        batch_size = 1024,
        num_proc = 8,
    )
    # 过滤过长的 SMILES
    dataset = dataset.filter(
        filter_long_smiles,
        batched = True,
        batch_size = 1024,
        num_proc = 8
    )
    dataset.save_to_disk('./dataset/build/smiles')
