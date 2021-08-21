from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.datamodule_utils import CustomSampler

from datamodules.chime_dataset import CHIMEDataset


class CHIMEDataModule(LightningDataModule):
    def __init__(self, batch_size, sizes_dataset, path_data, l_c, n_c, n_shards, **kwargs):
        super().__init__()

        self.dataset_args = {
            "path_data": path_data,
            "n_c": n_c,
            "l_c": l_c,
            "n_shards": n_shards,
        }
        self.batch_size = batch_size
        self.sizes_dataset = sizes_dataset
        self.n_shards = n_shards

        self.data_train = None
        self.data_valid = None
        self.data_predict = None

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        if stage == "fit":
            self.data_train = CHIMEDataset(
                split="train", size_dataset=self.sizes_dataset["train"], **self.dataset_args
            )
            self.data_valid = CHIMEDataset(
                split="valid", size_dataset=self.sizes_dataset["valid"], **self.dataset_args
            )
        else:
            self.data_predict = CHIMEDataset(
                split="test", size_dataset=self.sizes_dataset["test"], **self.dataset_args
            )

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["train"], n_shards=self.n_shards),
        )

    def predict_dataloader(self):
        """Return DataLoader for prediction."""

        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["test"], n_shards=self.n_shards),
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

        return None
