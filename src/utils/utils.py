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
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(
            f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>"
        )
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
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
import multiprocessing


class ParallelHelper:
    def __init__(
        self,
        f_task: object,
        data: list,
        data_allocation: object,
        num_workers: int = 4,
        desc=None,
    ):
        self.n_data = len(data)

        self.queue = multiprocessing.Queue()
        # self.pbar   = tqdm(total=self.n_data, desc=desc)

        self.jobs = list()
        for ith in range(num_workers):
            lo_bound = ith * self.n_data // num_workers
            hi_bound = (
                (ith + 1) * self.n_data // num_workers
                if ith < (num_workers - 1)
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

                # self.pbar.update()

        # self.pbar.close()

        for job in self.jobs:
            job.terminate()

        for job in self.jobs:
            job.join()

        return dataset