import glob
from random import sample

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NarrativeDataset(Dataset):
    def __init__(
        self,
        split,
        size_dataset,
        path_data,
        lc,
        nc,
        n_shards,
    ):

        self.split = split
        self.n_shards = n_shards
        self.size_dataset = size_dataset
        self.lc = lc
        self.nc = nc
        self.curent_ith_file = -1

        path_data = path_data.replace("[SPLIT]", split).replace("[SHARD]", "*")
        self.paths = sorted(glob.glob(path_data))

        self.q_ids = None
        self.c_ids = None
        self.a1_ids = None
        self.a2_ids = None

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
            "q_ids": self.q_ids[indx],
            "c_ids": self.c_ids[indx],
            "a1_ids": self.a1_ids[indx],
            "a2_ids": self.a2_ids[indx],
        }

    def _get_context(self, En, Hn):
        n_samples = min((len(En), self.nc))
        if self.split == "train":
            selects_Hn = int(n_samples * self.exchange_rate)
            selects_En = n_samples - selects_Hn

            return sample(En, selects_En) + sample(Hn, selects_Hn)

        return sample(Hn, n_samples)

    def read_datasetfile(self, path_file):
        df = pd.read_parquet(path_file)

        self.q_ids = []
        self.c_ids = []
        self.a1_ids = []
        self.a2_ids = []

        for entry in df.itertuples():
            q = entry.q_ids[(entry.q_ids != 101) & (entry.q_ids != 102)]
            a1 = entry.a1_ids[(entry.a1_ids != 101) & (entry.a1_ids != 102)]
            a2 = entry.a1_ids[(entry.a1_ids != 101) & (entry.a1_ids != 102)]
            cE = entry.c_E_ids[(entry.c_E_ids != 101) & (entry.c_E_ids != 102)]
            cE = np.reshape(cE, (-1, self.lc))
            cH = entry.c_H_ids[(entry.c_H_ids != 101) & (entry.c_H_ids != 102)]
            cH = np.reshape(cH, (-1, self.lc))

            self.q_ids.append(np.copy(q))
            self.a1_ids.append(np.copy(a1))
            self.a2_ids.append(np.copy(a2))

            c = np.zeros((self.nc, self.lc), dtype=int)
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

            self.c_ids.append(c)

    def switch_answerability(self):
        self.exchange_rate = min((1, self.exchange_rate + 0.15))
