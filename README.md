# NarrativeChime

This repo use CHIME model to deal with **NarrativeQA** dataset.

To run project, run:

> python run.py

By default, **Tensorboard** logger is used. To use **WandB** logger, run:

> python run.py +log=wandb

To run hyperparameter optimization, run:

> python run.py -m
