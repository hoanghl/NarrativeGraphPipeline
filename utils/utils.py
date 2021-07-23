import logging as log_std
import warnings
from typing import List, Sequence


import pytorch_lightning as pl
import rich.syntax
import rich.tree
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=log_std.INFO) -> log_std.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = log_std.getLogger(name)
    logger.setLevel(level)

    # this ensures all log_std levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("n_workers"):
            config.datamodule.n_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(
            f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>"
        )
        if config.datamodule.get("n_workers"):
            config.datamodule.n_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable log_std any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from log_std hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()


######################################################################################
# User-defined utils
######################################################################################
from importlib import import_module
from random import sample
import multiprocessing
import re

import torch
from torch.utils.data.sampler import Sampler
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def init_modules(configs: dict):
    modules = []

    for config in configs.values():
        target = config["_target_"]
        parts = target.split(".")
        mod = getattr(import_module(".".join(parts[:-1])), parts[-1])

        tmp = {**config}
        del tmp["_target_"]
        modules.append(mod(**tmp))

    return modules


class CustomSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, size_dataset, n_shards):
        self.size_dataset = size_dataset
        self.n_shards = n_shards

        self.indices = None
        self.shuffle()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return self.size_dataset

    def shuffle(self):
        size_shard = int(self.size_dataset / self.n_shards)

        shard_order = sample(range(self.n_shards), self.n_shards)

        self.indices = []
        for nth_shard in shard_order:
            start = size_shard * nth_shard
            end = (
                start + size_shard
                if nth_shard != max(shard_order)
                else self.size_dataset
            )
            indices = sample(range(start, end), end - start)

            self.indices.extend(indices)


class ParallelHelper:
    def __init__(
        self,
        f_task: object,
        data: list,
        data_allocation: object,
        n_workers,
        desc=None,
        show_bar=False,
    ):
        self.n_data = len(data)
        self.show_bar = show_bar

        self.queue = multiprocessing.Queue()
        if self.show_bar:
            self.pbar = tqdm(total=self.n_data, desc=desc)

        self.jobs = list()
        for ith in range(n_workers):
            lo_bound = ith * self.n_data // n_workers
            hi_bound = (
                (ith + 1) * self.n_data // n_workers
                if ith < (n_workers - 1)
                else self.n_data
            )

            p = multiprocessing.Process(
                target=f_task,
                args=(data_allocation(data, lo_bound, hi_bound), self.queue),
            )
            self.jobs.append(p)

    def launch(self) -> list:
        """
        Launch parallel process
        Returns: a list after running parallel task

        """
        dataset = []

        for job in self.jobs:
            job.start()

        cnt = 0
        while cnt < self.n_data:
            while not self.queue.empty():
                dataset.append(self.queue.get())
                cnt += 1

                if self.show_bar:
                    self.pbar.update()

        if self.show_bar:
            self.pbar.close()

        for job in self.jobs:
            job.terminate()

        for job in self.jobs:
            job.join()

        return dataset


def ipot(a1: torch.Tensor, a2: torch.Tensor, beta=2, max_iter=1000, L=1):
    """Calculate loss based on OT."""

    b, l_a, d_hid = a1.size()
    n = b * l_a

    # a1: [b, l_a, d_hid]
    # a2: [b, l_a, d_hid]

    a1, a2 = a1.view(-1, d_hid), a2.view(-1, d_hid)
    # [n, d_hid]

    # Calculate matrix C
    a1_norm = a1 / a1.norm(dim=1)[:, None]
    a2_norm = a2 / a2.norm(dim=1)[:, None]
    C = a1_norm @ a2_norm.transpose(0, 1)
    # [n, n]

    sigma = torch.ones((n, 1), device=a1.device) / n

    T = torch.ones((n, n), device=a1.device) / n ** 2
    # [n, n]
    A = torch.exp(-(C / beta))
    # [n, n]

    for _ in range(max_iter):
        Q = A * T
        # [n, n]

        for _ in range(L):
            d = 1 / n / (Q @ sigma)
            sigma = 1 / n / (Q.T @ d)

        d1 = torch.diag(d.squeeze(1))
        d2 = torch.diag(sigma.squeeze(1))
        T = d1 * Q * d2

    loss = torch.sum(T * C)

    return loss


def process_sent(sent):
    return re.sub(r"(\[PAD\]|\[CLS\]|\[SEP\]|\[UNK\]|\[MASK\])", "", sent).strip()


def get_scores(ref: list, pred, eps=10e-8):
    """Calculate metrics BLEU-1, BLEU4, METEOR and ROUGE_L.

    ref = [
        "the transcript is a written version of each day",
        "version of each day"
    ]
    pred= "a written version of"

    Args:
        ref (list): list of reference strings
        pred (str): string generated by model

    Returns:
        tuple: tuple of 4 scores
    """

    pred = process_sent(pred)
    ref = list(map(process_sent, ref))

    if pred == "":
        return 0, 0, 0, 0

    # Calculate BLEU score
    ref_ = [x.split() for x in ref]
    pred_ = pred.split()

    bleu_1 = sentence_bleu(ref_, pred_, weights=(1, 0, 0, 0))
    bleu_4 = sentence_bleu(ref_, pred_, weights=(0.25, 0.25, 0.25, 0.25))

    # Calculate METEOR
    meteor = meteor_score(ref, pred)

    # Calculate ROUGE-L
    scores = np.array(
        [Rouge().get_scores(ref_, pred, avg=True)["rouge-l"]["f"] for ref_ in ref]
    )
    rouge_l = np.mean(scores)

    return (
        bleu_1 if bleu_1 > eps else 0,
        bleu_4 if bleu_4 > eps else 0,
        meteor if meteor > eps else 0,
        rouge_l if rouge_l > eps else 0,
    )
