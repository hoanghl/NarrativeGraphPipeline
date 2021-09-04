# NarrativeGraphPipeline

## This repo contains code for incoming paper.

## 1. Prerequisites

I recommend using **conda**. To install conda environment, run the following:

>

**NarrativeQA** dataset version used is from **huggingface**.

## 2. How to run

Run the following:

> python run.py

### 3. How to tune

Run the following:

> python tune.py --multirun

### 4. Additional flags

- If using **WandB logger**, add the flag `+log=wandb`
- If using **multi GPUs**, add the flag `multigpu=True`
