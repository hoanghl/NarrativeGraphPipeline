import os
import sys
from typing import List

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

import utils.run_utils as utils

log = utils.get_logger(__name__)
dotenv.load_dotenv(override=True)


@hydra.main(config_path="./", config_name="configs.yaml")
def main(config: DictConfig):
    path_utils = os.environ.get("PATH_UTILS")
    config.PATH.utils = path_utils
    OmegaConf.resolve(config)

    if config.mode.startswith("debug"):
        config.callbacks = config.logger = None
        os.chdir(config.work_dir)

    if config.multigpu is True:
        config.trainer.precision = 32
        config.trainer.gpus = -1
        config.trainer.accelerator = "ddp"
        config.trainer.replace_sampler_ddp = True

    config.trainer.max_epochs = 10

    utils.extras(config)
    utils.print_config(config, resolve=True)

    log.info("########### Start tuning! ###########")

    # Init lightning loggers
    default_log = "tensorboard" if "log" not in config else config.log
    logger: List[LightningLoggerBase] = []
    if "logger" in config and config.logger is not None:
        for name, lg_conf in config.logger.items():
            if name != default_log:
                continue

            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    if not os.path.isfile(config.trainer.resume_from_checkpoint):
        log.info("=> No previous checkpoint specified/found. Start fresh training.")
        config.trainer.resume_from_checkpoint = None
    else:
        log.info("=> Previous checkpoint found. Resume training.")
    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")

    ## Check if checkpoint path is specified
    config.trainer.resume_from_checkpoint = None
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger, _convert_="partial")

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, num_gpus=trainer.num_gpus)

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        callbacks=None,
        datamodule=datamodule,
        trainer=trainer,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    log.info("Starting predicting!")
    bleu_1, bleu_4, meteor, rouge_l = trainer.predict(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        logger=logger,
    )

    return bleu_1


if __name__ == "__main__":
    main()
