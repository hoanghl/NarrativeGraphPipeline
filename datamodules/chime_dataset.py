from random import sample
import glob


from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class CHIMEDataset(Dataset):
    def __init__(
        self,
        split: str,
        size_dataset: int,
        path_data: str,
        l_c,
        n_c,
        n_shards,
    ):

        self.split = split
        self.n_shards = n_shards
        self.size_dataset = size_dataset
        self.l_c = l_c
        self.n_c = n_c
        self.curent_ith_file = -1

        path_data = path_data.replace("[SPLIT]", split).replace("[SHARD]", "*")
        self.paths = sorted(glob.glob(path_data))

        self.q = None
        self.c = None
        self.a = None

        self.exchange_rate = 0

    def __len__(self) -> int:
        return self.size_dataset

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        size_shard = self.size_dataset // self.n_shards

        ith_file = indx // size_shard
        indx = indx % size_shard

        if ith_file == self.n_shards:
            ith_file -= 1
            indx = indx + size_shard

        # Check nth file and reload dataset if needed
        if ith_file != self.curent_ith_file:
            self.curent_ith_file = ith_file

            # Reload dataset
            self.read_datasetfile(self.paths[self.curent_ith_file])

        return {
            "q": self.q[indx],
            "c": self.c[indx],
            "a": self.a[indx],
        }

    def _get_context(self, En, Hn):
        n_samples = min((len(En), self.n_c))
        if self.split == "train":
            selects_Hn = int(n_samples * self.exchange_rate)
            selects_En = n_samples - selects_Hn

            return sample(En, selects_En) + sample(Hn, selects_Hn)

        return sample(Hn, n_samples)

    def read_datasetfile(self, path_file):
        df = pd.read_parquet(path_file)

        self.q = []
        self.c = []
        self.a = []

        for entry in df.itertuples():
            q = entry.q_ids[(entry.q_ids != 101) & (entry.q_ids != 102)]
            a = entry.a1_ids[(entry.a1_ids != 101) & (entry.a1_ids != 102)]
            cE = entry.c_E_ids[(entry.c_E_ids != 101) & (entry.c_E_ids != 102)]
            cE = np.reshape(cE, (-1, self.l_c))
            cH = entry.c_H_ids[(entry.c_H_ids != 101) & (entry.c_H_ids != 102)]
            cH = np.reshape(cH, (-1, self.l_c))

            self.q.append(np.copy(q))
            self.a.append(np.copy(np.expand_dims(a, axis=0)))

            c = np.zeros((self.n_c, self.l_c), dtype=int)
            n_samples = cE.shape[0]
            if self.split == "train":

                ratio_Hn = int(n_samples * self.exchange_rate)
                indices_Hn = sample(range(n_samples), ratio_Hn)
                cH = cH[indices_Hn]

                ratio_En = n_samples - ratio_Hn
                indices_En = sample(range(n_samples), ratio_En)
                cE = cE[indices_En]

                c[:n_samples] = np.concatenate((cE, cH), axis=0)

            else:
                c[:n_samples] = cH

            self.c.append(c)

    def switch_answerability(self):
        self.exchange_rate = min((1, self.exchange_rate + 0.15))
