import argparse
import os
import pickle
from abc import ABCMeta, abstractmethod, ABC
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor

from ..logging.logger import Logger
from ..common.registry import registry
from ..dist import (broadcast_object_list, collect_results,
                           is_main_process)


class BaseMetric(metaclass=ABCMeta):
    """Base class for a metric.

    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.

    A subclass of class:`BaseMetric` should assign a meaningful value to the
    class attribute `default_prefix`. See the argument `prefix` for details.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
    """

    default_prefix: Optional[str] = None

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None) -> None:

        if collect_dir is not None and collect_device != 'cpu':
            raise ValueError('`collec_dir` could only be configured when '
                             "`collect_device='cpu'`")

        self._dataset_meta: Union[None, dict] = None
        # 依赖倒置，由runner -> evaluator(setup_state) -> metric 动态注入
        self.state = None
        self.collect_device = collect_device
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        self.collect_dir = collect_dir

        if self.prefix is None:
            print(
                'The prefix is not set in metric class '
                f'{self.__class__.__name__}.'
            )

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    @abstractmethod
    def process(self, data_batch: Any, data_samples: argparse.Namespace) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.'
            )

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = self.to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]

    def to_cpu(self, data: Any) -> Any:
        """Transfer all tensors to cpu."""
        if isinstance(data, Tensor):
            return data.to('cpu')
        elif isinstance(data, list):
            return [self.to_cpu(d) for d in data]
        elif isinstance(data, tuple):
            return tuple(self.to_cpu(d) for d in data)
        elif isinstance(data, dict):
            return {k: self.to_cpu(v) for k, v in data.items()}
        else:
            return data


@registry.register_metric("DumpResults")
class DumpResults(BaseMetric, ABC):
    """Dump model predictions to a pickle file for offline evaluation.

    Args:
        output_dir (str): Path of the dumped file. The data wil be stored
        as pickle file.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    def __init__(self,
                 output_dir: str,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)
        self.output_dir = output_dir.strip('/') + '/'

    @abstractmethod
    def process(self, data_batch: Any, predictions: dict) -> None:
        pass

    def compute_metrics(self, results: list) -> dict:
        """Save results to a pickle file with epoch/batch in filename if available."""
        folder = os.path.join(self.runner.cfg.run_dir, self.runner.cfg.run_name
                              , self.state.run_timestamp, self.output_dir)
        os.makedirs(folder, exist_ok=True)
        suffix = []

        current_epoch = self.state.current_epoch
        current_step = self.state.current_step
        if current_epoch is not None:
            suffix.append(f'epoch_{current_epoch}')
        if current_step is not None:
            suffix.append(f'_batch_{current_step}')
        out_file = f"{folder}/{''.join(suffix)}.pkl"

        with open(out_file, 'wb') as f:
            pickle.dump(results, f)

        Logger.get_current_instance().info(f'Results saved to {out_file}.')
        return {}
