from torch.utils.data import DataLoader
import pytorch_lightning as plt

from datamodules.utils import CustomSampler
from datamodules.dataset import NarrativeDataset

# from datamodules.preprocess import Preprocess


class NarrativeDataModule(plt.LightningDataModule):
    def __init__(
        self,
        path_data: str,
        path_pretrained: str,
        sizes_dataset: dict,
        batch_size,
        l_q,
        l_c,
        l_a,
        n_paras,
        n_workers,
        n_shards,
        **kwargs
    ):

        super().__init__()

        self.batch_size = batch_size
        self.l_q = l_q
        self.l_c = l_c
        self.l_a = l_a
        self.n_paras = n_paras
        self.path_data = path_data
        self.path_pretrained = path_pretrained
        self.n_workers = n_workers
        self.n_shards = n_shards
        self.sizes_dataset = sizes_dataset

        self.data_train = None
        self.data_test = None
        self.data_valid = None

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset_args = {
            "path_data": self.path_data,
            "path_pretrained": self.path_pretrained,
            "l_q": self.l_q,
            "l_c": self.l_c,
            "l_a": self.l_a,
            "n_paras": self.n_paras,
            "n_workers": self.n_workers,
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
            sampler=CustomSampler(self.sizes_dataset["train"], self.n_shards),
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""

        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["valid"], self.n_shards),
        )

    def test_dataloader(self):
        """Return DataLoader for test."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["test"], self.n_shards),
        )

    def predict_dataloader(self):
        """Return DataLoader for prediction."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["test"], self.n_shards),
        )

    def switch_answerability(self):
        self.data_train.switch_answerability()
