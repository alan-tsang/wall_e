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
            ans.append(None)
    return ans

def process_smiles(dataset):
    smiles_list = dataset['smi']
    smiles_list = standard_smiles(smiles_list)
    dataset['smiles'] = smiles_list
    return dataset

def filter_smiles(dataset):
    smiles_list = dataset['smiles']
    filtered_smiles = [smi for smi in smiles_list if smi is not None and len(smi) > 0]
    dataset['smiles'] = filtered_smiles
    return dataset

def tokenize(dataset):
    smiles_list = dataset['smiles']
    tokenized = tokenizer(
        smiles_list,
        padding=False,
        truncation=True,
        max_length=512
    )
    dataset['input_ids'] = tokenized['input_ids']
    dataset['attention_mask'] = tokenized['attention_mask']
    return dataset

if __name__  == '__main__':
    from transformers import AutoTokenizer
    model_checkpoint = './tokenizer/smiles'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    raw_dataset = load_dataset(
        "csv",
        data_files = "./dataset/raw/clean_smi.csv",
        column_names = ["smi"]
    )["train"]
    # raw_dataset = raw_dataset.select(range(1024))
    dataset = BaseMapDataset(
        data_source = raw_dataset,
        shuffle = True,
        split_ratios = (0.9999, 0.00005, 0.00005),
    )
    # (0.9999, 0.00005, 0.00005),
    # 首先处理 SMILES，但不移除列
    dataset = dataset.map(
        process_smiles,
        batch_size = 32,
        num_proc = 8,
        batched = True,
        remove_columns = ["smi"]
    )

    # 然后过滤无效的 SMILES
    dataset = dataset.filter(
        filter_smiles,
        batched = True,
        batch_size = 32,
        num_proc = 8
    )
    dataset.save_to_disk('./raw_smiles')

    dataset = dataset.map(
        tokenize,
        batched = True,
        batch_size = 32,
        num_proc = 8,
    )
    dataset.save_to_disk('./smiles')
