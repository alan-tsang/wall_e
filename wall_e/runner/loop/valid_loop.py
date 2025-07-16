import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .base_loop import BaseLoop
from ..util import move_data_to_device
from ...eval import Evaluator


class ValidLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Optional[Evaluator] = None,
                 shuffle = False,
                 fp16: bool = False
                 ):
        super().__init__(runner, dataloader, shuffle)

        self.evaluator = evaluator  # type: ignore
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""
        self.runner.before_valid()
        self.runner.model.eval()

        for idx, data_batch in enumerate(self.dataloader):
            data_batch = move_data_to_device(data_batch, self.runner.device)
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = {}
        if self.evaluator is not None:
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))  # type: ignore

        self.runner.after_valid()
        # self.runner.call_hook('after_val_epoch', metrics=metrics)
        # self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        # self.runner.call_hook(
        #     'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled = self.fp16):
            if hasattr(self.runner.model, "module"):
                # For DataParallel or DistributedDataParallel
                outputs = self.runner.model.module.valid_step(data_batch)
            else:
                outputs = self.runner.model.valid_step(data_batch)

        if self.evaluator is not None:
            self.evaluator.process(data_samples = outputs, data_batch = data_batch)
