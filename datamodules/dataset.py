from random import sample

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NarrativeDataset(Dataset):
    def __init__(self, split, path_data, nc, lc):

        self.nc = nc
        self.lc = lc

        self.q_ids = []
        self.a1_ids = []
        self.a2_ids = []
        self.c_ids = []
        self.c_masks = []

        self.exchange_rate = 0.5
        path_data = "/home/ubuntu/NarrativeGraph/example_71.parquet"
        self.read_datasetfile(path_data, split)

    def __len__(self) -> int:
        return len(self.q_ids)

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        return {
            "q_ids": self.q_ids[indx],
            "a1_ids": self.a1_ids[indx],
            "a2_ids": self.a2_ids[indx],
            "c_ids": self.c_ids[indx],
            "c_masks": self.c_masks[indx],
        }

    def read_datasetfile(self, path_data, split):

        df = pd.read_parquet(path_data.replace("[SPLIT]", split))

        for entry in df.itertuples():
            q = entry.q_ids[(entry.q_ids != 101) & (entry.q_ids != 102)]
            a1 = entry.a1_ids[(entry.a1_ids != 101) & (entry.a1_ids != 102)]
            a2 = entry.a1_ids[(entry.a1_ids != 101) & (entry.a1_ids != 102)]
            c_E_ids = entry.c_E_ids[(entry.c_E_ids != 101) & (entry.c_E_ids != 102)]
            c_E_ids = np.reshape(c_E_ids, (-1, self.lc))
            c_H_ids = entry.c_H_ids[(entry.c_H_ids != 101) & (entry.c_H_ids != 102)]
            c_H_ids = np.reshape(c_H_ids, (-1, self.lc))
            c_E_masks = np.reshape(entry.c_E_masks, (-1, self.lc + 2))[:, 2:]
            c_H_masks = np.reshape(entry.c_H_masks, (-1, self.lc + 2))[:, 2:]

            c_ids = np.zeros((self.nc, self.lc), dtype=int)
            c_masks = np.zeros((self.nc, self.lc), dtype=int)

            n_samples = c_E_ids.shape[0]
            if split == "train":
                ratio_Hn = int(n_samples * self.exchange_rate)
                indices_Hn = sample(range(n_samples), ratio_Hn)
                c_H_ids = c_H_ids[indices_Hn]
                c_H_masks = c_H_masks[indices_Hn]

                ratio_En = n_samples - ratio_Hn
                indices_En = sample(range(n_samples), ratio_En)
                c_E_ids = c_E_ids[indices_En]
                c_E_masks = c_E_masks[indices_En]

                c_ids[:n_samples] = np.concatenate((c_E_ids, c_H_ids), axis=0)
                c_masks[:n_samples] = np.concatenate((c_E_masks, c_H_masks), axis=0)

            else:
                c_ids[:n_samples] = c_H_ids
                c_masks[:n_samples] = c_H_masks

            self.q_ids.append(np.copy(q))
            self.a1_ids.append(np.copy(a1))
            self.a2_ids.append(np.copy(a2))
            self.c_ids.append(c_ids)
            self.c_masks.append(c_masks)

    # def switch_answerability(self):
    #     self.exchange_rate = min((1, self.exchange_rate + 0.25))
