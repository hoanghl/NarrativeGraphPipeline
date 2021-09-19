import pytorch_lightning as plt
from torch.utils.data import DataLoader
from utils.datamodule_utils import collate_fn

from datamodules.dataset import NarrativeDataset


class NarrativeDataModule(plt.LightningDataModule):
    def __init__(self, batch_size, lq, la, lc, nc, path_data, path_vocab):

        super().__init__()

        self.batch_size = batch_size
        self.dataset_args = {
            "path_data": path_data,
            "path_vocab": path_vocab,
            "lq": lq,
            "la": la,
            "lc": lc,
            "nc": nc,
        }

        self.data_train = None
        self.data_test = None
        self.data_valid = None

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        if stage == "fit":
            self.data_train = NarrativeDataset("train", **self.dataset_args)
            self.data_valid = NarrativeDataset("valid", **self.dataset_args)
        else:
            self.data_test = NarrativeDataset("test", **self.dataset_args)

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            dataset=self.data_train, batch_size=self.batch_size, collate_fn=collate_fn
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""

        return DataLoader(
            dataset=self.data_valid, batch_size=self.batch_size, collate_fn=collate_fn
        )

    def test_dataloader(self):
        """Return DataLoader for prediction."""

        return DataLoader(
            dataset=self.data_test, batch_size=self.batch_size, collate_fn=collate_fn
        )

    # def switch_answerability(self):
    #     self.data_train.switch_answerability()
