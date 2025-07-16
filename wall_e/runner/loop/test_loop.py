import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .base_loop import BaseLoop
from ..util import move_data_to_device
from ...eval import Evaluator

class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: DataLoader,
                 evaluator: Optional[Evaluator],
                 shuffle = False,
                 fp16: bool = False
                 ):
        super().__init__(runner, dataloader, shuffle)

        self.evaluator = evaluator  # type: ignore
        self.fp16 = fp16


    def run(self) -> dict:
        """Launch test."""
        self.runner.before_test()
        self.runner.model.eval()

        for idx, data_batch in enumerate(self.dataloader):
            data_batch = move_data_to_device(data_batch, self.runner.device)
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = {}
        if self.evaluator is not None:
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset)) # type: ignore

        self.runner.after_test()

        return metrics
    

    @torch.no_grad()
    def run_iter(self, idx, data_batch: dict[str, Sequence]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            if hasattr(self.runner.model, "module"):
                # For DataParallel or DistributedDataParallel
                outputs = self.runner.model.module.test_step(data_batch)
            else:
                outputs = self.runner.model.test_step(data_batch)

        if self.evaluator is not None:
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
