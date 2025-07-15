from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

from torch.utils.data import DataLoader


class BaseLoop(metaclass=ABCMeta):
    """Base loop class.

    All subclasses inherited from ``BaseLoop`` should overwrite the
    :meth:`run` method.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
    """

    def __init__(self, runner: 'Runner', dataloader: Union[DataLoader, Dict], shuffle) -> None:
        self._runner = runner
        self.dataloader = runner.wrap_dataloader(dataloader, shuffle = shuffle)

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""