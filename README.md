# Narrative Model

## This repo contains code for incoming paper.

## 1. Prerequisites

I recommend using **conda**. To install conda environment, run the following:

>

**NarrativeQA** dataset version used is from **huggingface**.

## 2. How to run

- Run the following to create training/validation data:

> python -m data_utils.preprocess

- Then run the following to start training:

> python run.py

## 3. Components

### 3.1. Data reading

Read stories and scripts from files, initially preprocess and decompose them into paragraphs.
