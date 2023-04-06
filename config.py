import yaml
import torch
import logging
import os

import torch

from memory import DiskMemory, Memory
from bots.custom_bot import BotBase



def build_params(params_dict=None, params_file=None):
    if params_dict is None:
        with open(params_file, "r") as f:
            params_dict = yaml.safe_load(f)

    replay_memory = Memory(
        size=params_dict["memory_size"],
        alpha=params_dict["alpha"],
        beta=params_dict["beta"],
        beta_decay=params_dict["beta_decay"],
        min_prob=params_dict["min_sample_prob"],
    )

    lr_decay_cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    lr_decay_fn = lambda optim: lr_decay_cls(optim, params_dict["n_restart"], 1, params_dict["lr_min"], verbose=False)

    new_dict = {"memory": replay_memory, "lr_decay": lr_decay_fn}

    to_copy = [
        "eps0", "lr0", "eps_decay", "discount", "n",
        "delta",
        "weight_decay","batch_size",
        "unit_model", "side_model", "number_model",
        "deck_names",
    ]
    for key in to_copy:
        new_dict[key] = params_dict[key]

    return new_dict


def build_options(opts_dict=None, opts_file=None):
    if opts_dict is None:
        with open(opts_file, "r") as f:
            opts_dict = yaml.safe_load(f)

    if "names" in opts_dict:
        shape_dict, dtype_dict = {}, {}
        for i, name in enumerate(opts_dict["names"]):
            shape_dict[name] = opts_dict["shapes"][i]
            dtype_dict[name] = opts_dict["dtypes"][i]
    else:
        shape_dict, dtype_dict = None, None

    disk_memory = DiskMemory(
        file=opts_dict["disk_memory"],
        data_transform=BotBase.exp_to_dict,
        shape_dict=shape_dict,
        dtype_dict=dtype_dict,
        max_size=opts_dict["max_size"] if "max_size" in opts_dict else None,
    )

    if opts_dict["logging"] == "print":
        log_fn = print
    elif opts_dict["logging"] is None:
        log_fn = lambda x: None
    else:
        os.makedirs(opts_dict["output"], exist_ok=True)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler_file = logging.FileHandler(opts_dict["logging"], mode=opts_dict["logging_mode"])
        logger.addHandler(handler_file)
        log_fn = logger.info

    new_dict = {
        "disk_memory": disk_memory,
        "device": torch.device(opts_dict["device"]),
        "logger": log_fn,
    }

    to_copy = [
        "checkpoint_frequency", "output"
    ]

    for key in to_copy:
        new_dict[key] = opts_dict[key]

    return new_dict
