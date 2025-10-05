"""
BaseDataset: based on HuggingFace Datasets, with some additional features.
"""

import json
import os.path
import types
import warnings
from typing import Union, Optional, Callable, List, Dict
from datasets import Dataset, DatasetDict, IterableDataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(
        self,
        data_source: Union[str, Dataset, DatasetDict],
        metadata: Optional[Dict] = None,
        *args,
        **kwargs
    ):
        """

        :param data_source: 数据路径/Hub名称/Dataset对象/DatasetDict对象
        :param process_fn:
        :param filter_fn:
        :param metadata:
        :param args:
        :param kwargs:
        """
        self.data_source = data_source
        self.metadata = metadata or {}
        self.dataset = None


    @abstractmethod
    def _set_dataset(self, *args, **kwargs):
        pass

    def get_batch_loader(self, split = None, transform_fn = None, **kwargs):
        """

        :param split:
        :param transform_fn:
        :param args:
        :param kwargs:
        :return:
        """
        if isinstance(self.dataset, DatasetDict):
            assert split is not None, "split should be specified when dataset is a DatasetDict."
            dataset = self.dataset[split]
        elif isinstance(self.dataset, (Dataset, IterableDataset)):
            dataset = self.dataset
        else:
            raise ValueError("Unsupported dataset type.")

        return self.__class__._get_batch_loader(dataset, transform_fn, **kwargs)

    @classmethod
    def _get_batch_loader(
            cls,
            dataset,
            transform_fn: Optional[Callable] = None,
            batch_size: int = 1,
            pin_memory: bool = False,
            sampler = None,
            num_workers: int = 0,
            collate_fn = None,
            shuffle: bool = False,
            **loader_kwargs
    ) -> DataLoader:
        if transform_fn is not None:
            dataset.set_transform(transform_fn)
        else:
            # 源码：相当于隐式dataset.set_transform("torch")
            dataset = dataset.with_format("torch")
        return DataLoader(
            dataset,
            batch_size = batch_size,
            pin_memory = pin_memory,
            sampler = sampler,
            num_workers = num_workers,
            collate_fn = collate_fn,
            shuffle = shuffle,
            **loader_kwargs
        )

    def map(self, fn, batched = False, batch_size = 1, num_proc = 8, remove_columns = None, **kwargs):
        return self._apply_operation('map', fn, batched, batch_size, num_proc, remove_columns, **kwargs)

    def filter(self, fn, batched = False, batch_size = 1, num_proc = 8, **kwargs):
        return self._apply_operation('filter', fn, batched, batch_size, num_proc, None, **kwargs)

    def _apply_operation(self, op_name, fn, batched = False, batch_size = 1, num_proc = 8, remove_columns = None,
                         **kwargs):
        """通用数据集操作分发器"""

        def process_ds(ds, op):
            base_params = {
                'function': fn,
                'batched': batched,
                'batch_size': batch_size,
                **kwargs
            }

            if isinstance(ds, Dataset):
                if op == 'filter':
                    return ds.filter(**base_params, num_proc = num_proc)
                return getattr(ds, op)(**base_params, num_proc = num_proc, remove_columns = remove_columns)

            if isinstance(ds, IterableDataset):
                # IterableDataset 不支持 remove_columns
                # 且 filter 操作不支持 num_proc
                if op == 'filter':
                    return ds.filter(**base_params)
                return ds.map(**base_params)

            raise ValueError(f"Unsupported dataset type: {type(ds)}")

        new_instance = self.__class__(
            data_source = self.data_source,
            metadata = self.metadata.copy()
        )

        if isinstance(self.dataset, (Dataset, IterableDataset)):
            new_instance.dataset = process_ds(self.dataset, op_name)
        elif isinstance(self.dataset, DatasetDict):
            new_instance.dataset = DatasetDict(
                {
                    split: process_ds(ds, op_name)
                    for split, ds in self.dataset.items()
                }
            )
        else:
            raise ValueError("Unsupported dataset type.")

        return new_instance


    def save_to_disk(self, path: str):
        os.makedirs(path, exist_ok=True)

        self.save_case(path)
        self.save_card(path)
        self.dataset.save_to_disk(path)


    def save_case(self, path: str):
        if isinstance(self.dataset, DatasetDict):
            for split in self.dataset.keys():
                case = str(self.dataset[split][0])
                with open(f"{path}/{split}_case.txt", "w") as f:
                    f.write(case)
        elif isinstance(self.dataset, Dataset):
            case = str(self.dataset[0])
            with open(f"{path}/dataset_case.txt", "w") as f:
                f.write(case)
        elif isinstance(self.dataset, IterableDataset):
            try:
                case = str(next(iter(self.dataset)))
            except StopIteration:
                warnings.warn("流式数据集未保存case data，因为现在已经没有数据，请检查数据源。")
            else:
                with open(f"{path}/dataset_case.txt", "w") as f:
                    f.write(case)


    def save_card(self, path: str):
        card = self.dataset_card
        with open(f"{path}/dataset_card.json", "w") as f:
            json.dump(card, f, indent=4)


    @property
    def dataset_card(self) -> Dict:
        return {
            **self.metadata,
        }


    @classmethod
    def from_cfg(cls, path: str, **kwargs) -> "BaseDataset":
        """兼容配置文件"""
        return cls(data_source = path, **kwargs)


    def push_to_hub(self, repo_id: str, **kwargs):
        """上传到HuggingFace Hub"""
        self.dataset.push_to_hub(repo_id, **kwargs)
