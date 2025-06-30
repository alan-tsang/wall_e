import json
import logging
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import torch
import yaml


def dump(data, filename, append_to_json = True, verbose = True):
    """
    Common i/o utility to handle saving data to various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    Specifically for .json, users have the option to either append (default)
    or rewrite by passing in Boolean value to append_to_json.
    """
    if verbose:
        logging.info(f"Saving data to file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with open(filename, "wb") as fopen:
            pickle.dump(data, fopen)
    elif file_ext == ".npy":
        with open(filename, "wb") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        if append_to_json:
            with open(filename, "a") as fopen:
                fopen.write(json.dumps(data, sort_keys = True) + "\n")
                fopen.flush()
        else:
            with open(filename, "w") as fopen:
                fopen.write(json.dumps(data, sort_keys = True) + "\n")
                fopen.flush()
    elif file_ext == ".yaml":
        with open(filename, "w") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    elif file_ext == ".pt":
        torch.save(data, filename)
        raise Exception(f"Saving {file_ext} is not supported yet")

    if verbose:
        logging.info(f"Saved data to file: {filename}")


def load(filename, mmap_mode = None, verbose = True, allow_pickle = False):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    if verbose:
        logging.info(f"Loading data from file: {filename}")

    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".txt":
        with open(filename, "r") as fopen:
            data = fopen.readlines()
    elif file_ext in [".pkl", ".pickle"]:
        with open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding = "latin1")
    elif file_ext == ".npy":
        if mmap_mode:
            try:
                with open(filename, "rb") as fopen:
                    data = np.load(
                        fopen,
                        allow_pickle = allow_pickle,
                        encoding = "latin1",
                        mmap_mode = mmap_mode,
                    )
            except ValueError as e:
                logging.info(
                    f"Could not mmap {filename}: {e}. Trying without g_pathmgr"
                )
                data = np.load(
                    filename,
                    allow_pickle = allow_pickle,
                    encoding = "latin1",
                    mmap_mode = mmap_mode,
                )
                logging.info("Successfully loaded without g_pathmgr")
            except Exception:
                logging.info("Could not mmap without  Trying without mmap")
                with open(filename, "rb") as fopen:
                    data = np.load(fopen, allow_pickle = allow_pickle, encoding = "latin1")
        else:
            with open(filename, "rb") as fopen:
                data = np.load(fopen, allow_pickle = allow_pickle, encoding = "latin1")
    elif file_ext == ".json":
        with open(filename, "r") as fopen:
            data = json.load(fopen)
    elif file_ext == ".yaml":
        with open(filename, "r") as fopen:
            data = yaml.load(fopen, Loader = yaml.FullLoader)
    elif file_ext == ".csv":
        with open(filename, "r") as fopen:
            data = pd.read_csv(fopen)
    elif file_ext == '.pt':
        data = torch.load(filename)
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    return data


def makedirs(path, verbose = False):
    os.makedirs(path, exist_ok = True)
    if verbose:
        print(f"创建文件夹： {path}")


def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        shutil.rmtree(dir)
    logging.info(f"Deleted contents of directory: {dir}")


def get_file_size(filename):
    """
    Given a file, get the size of file in MB
    """
    size_in_mb = os.path.getsize(filename) / float(1024 ** 2)
    return size_in_mb
