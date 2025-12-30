import os
from pathlib import Path
import wandb
from omegaconf import OmegaConf, DictConfig, DictConfig, DictConfig


def _load_recursive(path, loaded_files=None):
    """
    Recursively loads a configuration file and its defaults, handling nested structures.
    """
    if loaded_files is None:
        loaded_files = set()

    abs_path = Path(path).resolve()
    if abs_path in loaded_files:
        return OmegaConf.create()  # Avoid circular dependencies
    loaded_files.add(abs_path)

    if not abs_path.exists():
        return OmegaConf.create()

    main_cfg = OmegaConf.load(abs_path)
    config_dir = abs_path.parent

    # Start with an empty config that we will build up
    merged_cfg = OmegaConf.create()

    # Process the defaults list first
    if 'defaults' in main_cfg:
        for default in main_cfg.defaults:
            if default == '_self_':
                # In Hydra, _self_ is a marker. The file's contents are merged at its position.
                # To match the user's expectation of _self_ being a base, we merge it first.
                self_cfg = OmegaConf.masked_copy(main_cfg, list(main_cfg.keys() - {'defaults'}))
                merged_cfg = OmegaConf.merge(merged_cfg, self_cfg)
                continue

            if isinstance(default, DictConfig):
                for key, value in default.items():
                    default_path = config_dir / key / f"{value}.yaml"
                    # Recursive call to handle nested defaults
                    default_cfg = _load_recursive(default_path, loaded_files)
                    # The new config (default_cfg) overrides the existing one (merged_cfg)
                    default_cfg = OmegaConf.create({key: default_cfg})
                    merged_cfg = OmegaConf.merge(merged_cfg, default_cfg)
    else:
        # If no 'defaults' list, the file content is the config
        merged_cfg = main_cfg

    return merged_cfg


def load_cfg(path):
    """
    Load configuration from a file, supporting Hydra-style recursive defaults.
    """
    # Recursively load the base configuration and all its defaults
    base_cfg = _load_recursive(path)

    # Merge CLI overrides, which have the highest precedence over file-based configs
    cli_cfg = OmegaConf.from_cli()

    # Final merge to ensure CLI overrides everything
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # DeepSpeed config handling remains separate as it's often a CLI arg
    ds_config_path = cfg.get("training.ds_config", None)
    if ds_config_path and os.path.exists(ds_config_path):
        ds_cfg = OmegaConf.load(ds_config_path)
        cfg = OmegaConf.merge(cfg, ds_cfg)

    return cfg
