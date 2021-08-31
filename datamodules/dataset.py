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
        path_data,
        lc,
        nc,
    ):

        self.split = split
        self.lc = lc
        self.nc = nc

        self.q_ids = []
        self.c_ids = []
        self.a1_ids = []
        self.a2_ids = []

        self.exchange_rate = 0

        self.read_datasetfile(path_data, split)

    def __len__(self) -> int:
        return len(self.q_ids)

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        return {
            "q_ids": self.q_ids[indx],
            "c_ids": self.c_ids[indx],
            "a1_ids": self.a1_ids[indx],
            "a2_ids": self.a2_ids[indx],
        }

    def read_datasetfile(self, path_data, split):
        df = pd.read_parquet(path_data.replace("[SPLIT]", split))

        for entry in df.itertuples():
            self.q_ids.append(np.copy(entry.q_ids))
            self.a1_ids.append(np.copy(entry.a1_ids))
            self.a2_ids.append(np.copy(entry.a2_ids))

            cE = np.reshape(entry.c_E_ids, (-1, self.lc))
            cH = np.reshape(entry.c_H_ids, (-1, self.lc))
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
