import pytorch_lightning as plt
from torch.utils.data import DataLoader

from datamodules.dataset import NarrativeDataset


class NarrativeDataModule(plt.LightningDataModule):
    def __init__(self, batch_size, lc, nc, path_data):

        super().__init__()

        self.batch_size = batch_size
        self.lc = lc
        self.nc = nc
        self.path_data = path_data

        self.data_train = None
        self.data_test = None
        self.data_valid = None

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset_args = {"path_data": self.path_data, "lc": self.lc, "nc": self.nc}
        if stage == "fit":
            self.data_train = NarrativeDataset("train", **dataset_args)
            self.data_valid = NarrativeDataset("valid", **dataset_args)
        else:
            self.data_test = NarrativeDataset("test", **dataset_args)

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""

        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
        )

    def predict_dataloader(self):
        """Return DataLoader for prediction."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
        )

    def switch_answerability(self):
        self.data_train.switch_answerability()
