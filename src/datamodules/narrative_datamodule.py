from torch.utils.data import DataLoader
import pytorch_lightning as plt

from src.datamodules.utils import CustomSampler
from src.datamodules.dataset import NarrativeDataset

# from src.datamodules.preprocess import Preprocess


class NarrativeDataModule(plt.LightningDataModule):
    def __init__(
        self,
        path_raw_data: str,
        path_processed_contx: str,
        path_data: str,
        path_bert: str,
        sizes_dataset: dict,
        batch_size,
        l_q,
        l_c_processing,
        l_c,
        l_a,
        n_paras,
        num_workers,
        **kwargs
    ):

        super().__init__()

        self.batch_size = batch_size
        self.l_q = l_q
        self.l_c_processing = l_c_processing
        self.l_c = l_c
        self.l_a = l_a
        self.n_paras = n_paras
        self.path_raw_data = path_raw_data
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.path_bert = path_bert
        self.num_workers = num_workers
        self.sizes_dataset = sizes_dataset

        self.data_train = None
        self.data_test = None
        self.data_valid = None

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset_args = {
            "path_data": self.path_data,
            "path_bert": self.path_bert,
            "l_q": self.l_q,
            "l_c": self.l_c,
            "l_a": self.l_a,
            "n_paras": self.n_paras,
            "num_worker": self.num_workers,
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
            sampler=CustomSampler(self.sizes_dataset["train"]),
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""

        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["valid"]),
        )

    def test_dataloader(self):
        """Return DataLoader for test."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["test"]),
        )

    def predict_dataloader(self):
        """Return DataLoader for prediction."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            sampler=CustomSampler(self.sizes_dataset["test"]),
        )

    def switch_answerability(self):
        self.data_train.switch_answerability()
