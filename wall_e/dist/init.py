"""
adapted from
https://github.com/salesforce/LAVIS/blob/main/lavis/common/dist_utils.py
"""
import argparse
import datetime
import functools
import os

import torch
import torch.distributed as dist


def setup_print_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args=None):
    if args is None:
        args = argparse.Namespace(dist_url = "env://")
    elif os.environ.get("MASTER_ADDR", None) is not None:
        args.dist_url = "tcp://{}:{}".format(
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]
        )
    else:
        assert "dist_url" in args, "dist_url must be specified"


    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return False

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"

    try:
        from ..logging.logger import Logger
        logger = Logger.get_current_instance()
    except Exception as e:
        print(
            "Distributed Launching: init (rank {}, world {}): {}".format(
                args.rank, args.world_size, args.dist_url
            ),
            flush = True,
        )
    else:
        logger.info(
            "Distributed Launching: init (rank {}, world {}): {}".format(
                args.rank, args.world_size, args.dist_url
            )
        )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    torch.distributed.barrier()
    setup_print_for_distributed(args.rank == 0)


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
