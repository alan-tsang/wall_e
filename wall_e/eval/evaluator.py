import ray
from typing import Any, Iterator, List, Optional, Sequence, Union


from ..dataset.utils import pseudo_collate, default_collate
from ..common.registry import registry
from .base_metric import BaseMetric



class Evaluator:
    """Wrapper class to compose multiple :class:`BaseMetric` instances.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
    """

    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        self._dataset_meta: Optional[dict] = None
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
        self.metrics: List[BaseMetric] = []
        for metric in metrics:
            if isinstance(metric, dict):
                metric_class = registry.get_evaluator(metric)["type"]
                self.metrics.append(metric_class(**metric))
            else:
                self.metrics.append(metric)

    def setup_state(self, state, runner):
        for metric in self.metrics:
            metric.state = state
            metric.runner = runner

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the evaluator and it's metrics."""
        self._dataset_meta = dataset_meta
        for metric in self.metrics:
            metric.dataset_meta = dataset_meta

    def process(self,
                data_samples,
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """
        for metric in self.metrics:
            metric.process(data_batch, data_samples)

    @staticmethod
    def register_metrics(metrics: dict):
        # 判断是否为omegaconf
        for key, value in metrics.items():
            registry.register(f"metric.{key}", value)


    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """
        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)

        self.register_metrics(metrics)

        return metrics


    def offline_evaluate(self,
                         data_samples: Sequence,
                         data: Optional[Sequence] = None,
                         chunk_size: int = 1):
        """Offline evaluate the dumped predictions on the given data .

        given data is a Sequence.
        if it is a list of dict:
        [
            {'x1': x11, 'x2': x21, 'x3': x31……},
            {'x1': x12, 'x2': x22, 'x3': x32……}
        ]
        pesudo_collate will make it
        [
            {
                'x1': [x11, x12……]
                'x2': [x21, x22……]
            }
        ]
        if it is a list of tensor:
        [
            [x11, x21, x31……],
            [x12, x22, x32……]
        ]
        pesudo_collate will make it
        [
            [(x11, x12……), (x21, x22……), (x31, x32……)]
        ]

        Args:
            data_samples (Sequence): All predictions and ground truth of the
                model and the validation set.
            data (Sequence, optional): All data of the validation set.
            chunk_size (int): The number of data samples and predictions to be
                processed in a batch.
        """

        # support chunking iterable objects
        def get_chunks(seq: Iterator, chunk_size=1):
            stop = False
            while not stop:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(next(seq))
                    except StopIteration:
                        stop = True
                        break
                if chunk:
                    yield chunk

        if data is not None:
            assert len(data_samples) == len(data), (
                'data_samples and data should have the same length, but got '
                f'data_samples length: {len(data_samples)} '
                f'data length: {len(data)}')
            data = get_chunks(iter(data), chunk_size)

        size = 0
        for output_chunk in get_chunks(iter(data_samples), chunk_size):
            if data is not None:
                data_chunk = pseudo_collate(next(data))  # type: ignore
            else:
                data_chunk = None
            size += len(output_chunk)
            output_chunk = pseudo_collate(output_chunk)
            self.process(output_chunk, data_chunk)
        return self.evaluate(size)
