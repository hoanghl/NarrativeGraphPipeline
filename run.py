from typing import List
import sys
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
import dotenv

import utils.run_utils as utils

log = utils.get_logger(__name__)
dotenv.load_dotenv(override=True)


@hydra.main(config_path="./", config_name="configs.yaml")
def main(config: DictConfig):
    path_utils = os.environ.get("PATH_UTILS")
    config.PATH.utils = path_utils
    OmegaConf.resolve(config)

    if "mode" not in config:
        log.error("Mode not specified in config.")
        sys.exit(1)
    assert config.mode in [
        "train",
        "predict",
        "debug_train",
        "debug_predict",
    ], f"Mode incorrect. Must be 'train'/'predict/debug_train/debug_predict', not {config.mode}"

    if config.mode.startswith("debug"):
        config.callbacks = config.logger = None
        os.chdir(config.work_dir)

    utils.extras(config)
    utils.print_config(config, resolve=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model_kwargs = {"datamodule": datamodule}
    model: LightningModule = hydra.utils.instantiate(config.model, **model_kwargs)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config and config.callbacks is not None:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

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

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")

    ## Check if checkpoint path is specified
    path_resume = config.trainer.resume_from_checkpoint
    if not os.path.isfile(config.trainer.resume_from_checkpoint):
        log.info("=> No previous checkpoint specified/found. Start fresh training.")
        config.trainer.resume_from_checkpoint = None
    else:
        log.info("=> Previous checkpoint found. Resume training.")

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.mode.endswith("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
    else:
        log.info("Starting predicting!")
        trainer.predict(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


if __name__ == "__main__":
    main()
