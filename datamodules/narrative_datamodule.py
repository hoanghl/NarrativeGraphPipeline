from torch.utils.data import DataLoader
import pytorch_lightning as plt

from utils.datamodule_utils import CustomSampler
from datamodules.dataset import NarrativeDataset


class NarrativeDataModule(plt.LightningDataModule):
    def __init__(
        self,
        batch_size,
        sizes_dataset,
        l_c,
        n_c,
        n_shards,
        path_data,
        **kwargs,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.l_c = l_c
        self.n_c = n_c
        self.path_data = path_data
        self.n_shards = n_shards
        self.sizes_dataset = sizes_dataset

        self.data_train = None
        self.data_test = None
        self.data_valid = None

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset_args = {
            "path_data": self.path_data,
            "l_c": self.l_c,
            "n_c": self.n_c,
            "n_shards": self.n_shards,
        }
        if stage == "fit":
            self.data_train = NarrativeDataset(
                "train", size_dataset=self.sizes_dataset["train"], **dataset_args
            )
            self.data_valid = NarrativeDataset(
                "valid", size_dataset=self.sizes_dataset["valid"], **dataset_args
            )
        else:
            self.data_test = NarrativeDataset(
                "test", size_dataset=self.sizes_dataset["test"], **dataset_args
            )

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["train"], n_shards=self.n_shards),
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""

        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["valid"], n_shards=self.n_shards),
        )

    def test_dataloader(self):
        """Return DataLoader for test."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["test"], n_shards=self.n_shards),
        )

    def predict_dataloader(self):
        """Return DataLoader for prediction."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["test"], n_shards=self.n_shards),
        )

    def switch_answerability(self):
        self.data_train.switch_answerability()
