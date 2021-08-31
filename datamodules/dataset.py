import glob
from random import sample

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NarrativeDataset(Dataset):
    def __init__(self, split, path_data, nc, lc):

        self.split = split
        self.nc = nc
        self.lc = lc

        self.q_ids = []
        self.a1_ids = []
        self.a1_masks = []
        self.a2_ids = []
        self.a2_masks = []
        self.c_ids = []

        self.exchange_rate = 0.75

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
            "a1_masks": self.a1_masks[indx],
            "a2_masks": self.a2_masks[indx],
            "c_ids": self.c_ids[indx],
        }

    def read_datasetfile(self, path_data, split):
        def get_masks(ids):
            masks = np.zeros(ids.shape)
            masks[: ids[ids != 0].shape[0]] = 1

            return masks

        df = pd.read_parquet(path_data.replace("[SPLIT]", split))

        for entry in df.itertuples():
            q = entry.q_ids[(entry.q_ids != 101) & (entry.q_ids != 102)]
            a1 = entry.a1_ids[(entry.a1_ids != 101) & (entry.a1_ids != 102)]
            a2 = entry.a1_ids[(entry.a1_ids != 101) & (entry.a1_ids != 102)]
            a1_mask = entry.a1_masks[(entry.a1_masks != 101) & (entry.a1_masks != 102)]
            a2_mask = get_masks(entry.a2_ids)
            a2_mask = a2_mask[(a2_mask != 101) & (a2_mask != 102)]
            cE = entry.c_E_ids[(entry.c_E_ids != 101) & (entry.c_E_ids != 102)]
            cE = np.reshape(cE, (-1, self.lc))
            cH = entry.c_H_ids[(entry.c_H_ids != 101) & (entry.c_H_ids != 102)]
            cH = np.reshape(cH, (-1, self.lc))

            self.q_ids.append(np.copy(q))
            self.a1_ids.append(np.copy(a1))
            self.a2_ids.append(np.copy(a2))
            self.a1_masks.append(np.copy(a1_mask))
            self.a2_masks.append(np.copy(a2_mask))

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

    # def switch_answerability(self):
    #     self.exchange_rate = min((1, self.exchange_rate + 0.25))
