import os

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import OmegaConf
import dotenv

from data_utils.narrative_datamodule import NarrativeDataModule
from model_utils.narrative_model import NarrativeModel
import utils.utils as utils


dotenv.load_dotenv(override=True)
log = utils.get_logger()


def run():
    config = OmegaConf.load("./configs.yaml")

    # Replace path_utils to value read from ENV
    path_utils = os.environ.get("NARRATIVE_UTILS")
    config.PATH.utils = path_utils

    OmegaConf.resolve(config)

    assert config.mode in [
        "train",
        "predict",
    ], "Mode incorrect. Must be 'train'/'predict'"

    ## Do some miscellaneous things
    utils.extras(config)
    utils.print_config(config, resolve=True)

    log.info("Instantiating datamodule, model, callbacks, logger and trainer")
    datamodule: LightningDataModule = NarrativeDataModule(**dict(config.datamodule))
    model: LightningModule = NarrativeModel(**dict(config.model), datamodule=datamodule)
    callbacks = utils.init_modules(dict(config.callbacks))
    logger = utils.init_modules(dict(config.logger))

    if not os.path.isfile(config.trainer.resume_from_checkpoint):
        if config.mode == "predict":
            raise FileNotFoundError("In 'predict' mode and no checkpoint found.")
        log.info("=> No previous checkpoint specified/found. Start fresh training.")
        config.trainer.resume_from_checkpoint = None
    else:
        log.info("=> Previous checkpoint found. Resume training.")

    trainer: Trainer = Trainer(
        callbacks=callbacks, logger=logger, **dict(config.trainer)
    )

    # Train/Predict the model
    if config.mode == "train":
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
    else:
        log.info("Starting predicting!")
        trainer.predict(model=model, datamodule=datamodule)


if __name__ == "__main__":
    run()
