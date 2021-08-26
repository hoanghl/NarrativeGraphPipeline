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
        size_dataset,
        n_c,
        lc,
        n_shards,
    ):

        self.split = split
        self.n_c = n_c
        self.lc = lc
        self.n_shards = n_shards
        self.size_dataset = size_dataset

        path_data = path_data.replace("[SPLIT]", split).replace("[SHARD]", "*")
        self.paths = sorted(glob.glob(path_data))

        self.curent_ith_file = -1

        self.q_ids = None
        self.q_masks = None
        self.a1_ids = None
        self.a1_masks = None
        self.a2_ids = None
        self.c_ids = None
        self.c_masks = None

        self.exchange_rate = 0.5

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
            "q_masks": self.q_masks[indx],
            "a1_ids": self.a1_ids[indx],
            "a2_ids": self.a2_ids[indx],
            "a1_masks": self.a1_masks[indx],
            "c_ids": self.c_ids[indx],
            "c_masks": self.c_masks[indx],
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

        self.q_ids = []
        self.q_masks = []
        self.a1_ids = []
        self.a1_masks = []
        self.a2_ids = []
        self.c_ids = []
        self.c_masks = []

        for entry in df.itertuples():
            self.q_ids.append(np.copy(entry.q_ids))
            self.q_masks.append(np.copy(entry.q_masks))
            self.a1_ids.append(np.copy(entry.a1_ids))
            self.a1_masks.append(np.copy(entry.a1_masks))
            self.a2_ids.append(np.copy(entry.a2_ids))

            c_E_ids = np.copy(np.reshape(entry.c_E_ids, (-1, self.lc)))
            c_E_masks = np.copy(np.reshape(entry.c_E_masks, (-1, self.lc)))
            c_H_ids = np.copy(np.reshape(entry.c_H_ids, (-1, self.lc)))
            c_H_masks = np.copy(np.reshape(entry.c_H_masks, (-1, self.lc)))

            c_ids = np.zeros((self.n_c, self.lc), dtype=int)
            c_masks = np.zeros((self.n_c, self.lc), dtype=int)

            n_samples = c_E_ids.shape[0]
            if self.split == "train":

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

            self.c_ids.append(c_ids)
            self.c_masks.append(c_masks)

    # def switch_answerability(self):
    #     self.exchange_rate = min((1, self.exchange_rate + 0.25))
