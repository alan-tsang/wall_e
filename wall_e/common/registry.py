"""
adapted from https://github.com/salesforce/LAVIS/blob/main/lavis/common/registry.py
"""
from typing import Optional
from ..model import BaseModel

class Registry:
    mapping = {
        "model_name_mapping": {},
        "lr_scheduler_name_mapping": {},
        "runner_name_mapping": {},
        "metric_name_mapping": {},
        "dataset_name_mapping": {},
        "callback_name_mapping": {},
        "state": {},
        "paths": {},
    }

    @classmethod
    def register_model(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from pipeline.common.registry import registry
        """

        def wrap(model_cls):
            from torch import nn
            from ..model import BaseModel

            assert issubclass(
                model_cls, (BaseModel, nn.Module)
            ), "All models must inherit BaseModel or nn.Module class"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_metric(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from pipeline.common.registry import registry
        """

        def wrap(metric_cls):
            from ..eval.base_metric import BaseMetric

            assert issubclass(
                metric_cls, BaseMetric
            ), "All Metrics must inherit BaseMetric class"
            if name in cls.mapping["metric_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["metric_name_mapping"][name]
                    )
                )
            cls.mapping["metric_name_mapping"][name] = metric_cls
            return metric_cls

        return wrap

    @classmethod
    def register_dataset(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from pipeline.common.registry import registry
        """

        def wrap(dataset_cls):
            from ..dataset.base_dataset import BaseDataset

            assert issubclass(
                dataset_cls, BaseDataset
            ), "All datasets must inherit BaseDataset class"
            if name in cls.mapping["dataset_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["dataset_name_mapping"][name]
                    )
                )
            cls.mapping["dataset_name_mapping"][name] = dataset_cls
            return dataset_cls

        return wrap


    @classmethod
    def register_callback(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

        """
        def wrap(callback_cls):
            from ..callback.base_callback import BaseCallBack
            assert issubclass(
                callback_cls, BaseCallBack
            ), "All CallBack must inherit BaseCallBack class"

            if name in cls.mapping["callback_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["callback_name_mapping"][name]
                    )
                )
            cls.mapping["callback_name_mapping"][name] = callback_cls
            return callback_cls

        return wrap



    @classmethod
    def register_lr_scheduler(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from pipeline.common.registry import registry
        """

        def wrap(lr_sched_cls):
            if name in cls.mapping["lr_scheduler_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["lr_scheduler_name_mapping"][name]
                    )
                )
            cls.mapping["lr_scheduler_name_mapping"][name] = lr_sched_cls
            return lr_sched_cls

        return wrap

    @classmethod
    def register_runner(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from pipeline.common.registry import registry
        """

        def wrap(runner_cls):
            from ..runner.runner import Runner
            assert issubclass(
                runner_cls, Runner
            ), "All runners must inherit Runner class"
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["runner_name_mapping"][name]
                    )
                )
            cls.mapping["runner_name_mapping"][name] = runner_cls
            return runner_cls

        return wrap



    @classmethod
    def register_path(cls, name, path):
        r"""Register a path to registry with key 'name'

        Args:
            name: Key with which the path will be registered.

        Usage:

            from pipeline.common.registry import registry
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name' to the state dict

        Args:
            name: Key with which the item will be registered.

        Usage::

            from pipeline.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj


    @classmethod
    def get_model_class(cls, name) -> Optional[BaseModel]:
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_dataset_class(cls, name):
        return cls.mapping["dataset_name_mapping"].get(name, None)

    @classmethod
    def get_metric_class(cls, name):
        return cls.mapping["metric_name_mapping"].get(name, None)

    @classmethod
    def get_lr_scheduler_class(cls, name):
        return cls.mapping["lr_scheduler_name_mapping"].get(name, None)

    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)

    @classmethod
    def get_callback_class(cls, name):
        return cls.mapping["callback_name_mapping"].get(name, None)

    @classmethod
    def list_runners(cls):
        return sorted(cls.mapping["runner_name_mapping"].keys())

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["dataset_name_mapping"].keys())

    @classmethod
    def list_callbacks(cls):
        return sorted(cls.mapping["callback_name_mapping"].keys())

    @classmethod
    def list_metrics(cls):
        return cls.mapping["metric_name_mapping"].keys()

    @classmethod
    def list_lr_schedulers(cls):
        return sorted(cls.mapping["lr_scheduler_name_mapping"].keys())

    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            # NOTE: This is a hack to make sure that get values from pydantic BaseModel
            try:
                value = value.get(subname, default)
            except:
                value = dict(value)
                value = value.get(subname, default)

            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
