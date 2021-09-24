import pytorch_lightning as plt
from torch.utils.data import DataLoader
from utils.datamodule_utils import Batch, Vocab, load_data

from datamodules.dataset import SummDataset, TextDataset


class NarrativeDataModule(plt.LightningDataModule):
    def __init__(self, batch_size, lq, la, lc, nc, path_data, path_vocab):

        super().__init__()

        self.batch_size = batch_size
        self.path_data = path_data
        self.lq = lq
        self.lc = lc
        self.la = la
        self.nc = nc

        self.data_train = None
        self.data_test = None
        self.data_valid = None
        self.vocab = Vocab.from_json(path_vocab)

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        if stage == "fit":
            self.data_train = self.load_dataset("train")
            self.data_valid = self.load_dataset("valid")

        else:
            self.data_test = self.load_dataset("test")

    def load_dataset(self, split):
        q, c, a = load_data(self.path_data.replace("[SPLIT]", split))

        q = TextDataset(q, self.lq)
        c = TextDataset(c, self.lc, is_context=True)
        a = TextDataset(a, self.la)

        return SummDataset(q, c, a, vocab=self.vocab)

    def train_dataloader(self):
        return DataLoaderBuilder()(
            dataset=self.data_train,
            vocab=self.vocab,
            batch_size=self.batch_size,
            max_decode=self.la,
            is_train=True,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoaderBuilder()(
            dataset=self.data_valid,
            vocab=self.vocab,
            batch_size=1,
            max_decode=self.la,
            is_train=False,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoaderBuilder()(
            dataset=self.data_test,
            vocab=self.vocab,
            batch_size=1,
            max_decode=self.la,
            is_train=False,
            num_workers=1,
        )

    # def switch_answerability(self):
    #     self.data_train.switch_answerability()


class DataLoaderBuilder:
    def __init__(self) -> None:
        self.vocab, self.max_decode = None, None

    def collate_fn(self, data):
        return Batch(data=data, vocab=self.vocab, max_decode=self.max_decode)

    def __call__(self, dataset, vocab, batch_size, max_decode, is_train, num_workers):
        self.vocab, self.max_decode = vocab, max_decode

        shuffle = True if is_train else False
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
        )
        return data_loader
