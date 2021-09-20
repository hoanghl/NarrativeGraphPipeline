import pickle
from random import sample

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.datamodule_utils import Vocab


class NarrativeDataset(Dataset):
    def __init__(self, split, path_data, path_vocab, lq, la, lc, nc):

        self.lc = lc
        self.nc = nc
        self.lq = lq
        self.la = la
        self.vocab = Vocab(path_vocab)

        self.q_ids = []
        self.q_masks = []
        self.c_ids = []
        self.c_ids_ext = []
        self.c_masks = []
        self.dec_inp1_ids = []
        self.dec_inp2_ids = []
        self.dec_trg1_ids = []
        self.dec_trg2_ids = []
        self.trg1_txt = []
        self.trg2_txt = []
        self.oovs = []

        self.exchange_rate = 0.6

        self.read_datasetfile(path_data, split)

    def __len__(self) -> int:
        return len(self.q_ids)

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        return {
            "q_ids": self.q_ids[indx],
            "q_masks": self.q_masks[indx],
            "c_ids": self.c_ids[indx],
            "c_ids_ext": self.c_ids_ext[indx],
            "c_masks": self.c_masks[indx],
            "dec_inp1_ids": self.dec_inp1_ids[indx],
            "dec_inp2_ids": self.dec_inp2_ids[indx],
            "dec_trg1_ids": self.dec_trg1_ids[indx],
            "dec_trg2_ids": self.dec_trg2_ids[indx],
            "trg1_txt": self.trg1_txt[indx],
            "trg2_txt": self.trg2_txt[indx],
            "oovs": self.oovs[indx],
        }

    def read_datasetfile(self, path_data, split):
        with open(path_data.replace("[SPLIT]", split), "rb") as f:
            data = pickle.load(f)

        for entry in data:
            q = self.vocab.stoi(entry["q"])
            q = self._trunc(q, self.lq)
            q_ids, q_masks = self._pad(q, self.lq)

            cE, cH = entry["cE"], entry["cH"]
            n_samples = len(cE)
            if split == "train":
                ratio_Hn = int(n_samples * self.exchange_rate)
                indices_Hn = sample(range(n_samples), ratio_Hn)
                cH = [cH[i] for i in indices_Hn]

                ratio_En = n_samples - ratio_Hn
                indices_En = sample(range(n_samples), ratio_En)
                cE = [cE[i] for i in indices_En]

                c = np.concatenate(cE + cH)
            else:
                c = np.concatenate(cH)

            oovs = {}
            c_ids = self.vocab.stoi(c)
            c_ids_ext = []
            for i, (tok_, id_) in enumerate(zip(c, c_ids)):
                if id_ == self.vocab.unk_id:
                    oovs[tok_] = i
                    c_ids_ext.append(self.lc + i)
                else:
                    c_ids_ext.append(id_)
            c_ids = self._trunc(c_ids, self.lc)
            c_ids, c_masks = self._pad(c_ids, self.lc)
            c_ids_ext, _ = self._pad(c_ids_ext, self.lc)

            a1 = self.vocab.stoi(entry["a1"])
            a1_ = a1.copy()
            for i, (tok_, id_) in enumerate(zip(entry["a1"], a1)):
                if id_ == self.vocab.unk_id and tok_ in oovs:
                    a1[i] = len(self.vocab) + oovs[tok_]
            dec_inp1_ids, _, dec_trg1_ids = self._add_decoding_toks(a1, a1_)
            trg1_txt = entry["a1"].tolist()[: self.la]
            trg1_txt = trg1_txt + [self.vocab.pad] * (self.la - len(trg1_txt))

            a2 = self.vocab.stoi(entry["a2"])
            a2_ = a2.copy()
            for i, (tok_, id_) in enumerate(zip(entry["a2"], a2)):
                if id_ == self.vocab.unk_id and tok_ in oovs:
                    a2[i] = len(self.vocab) + oovs[tok_]
            dec_inp2_ids, _, dec_trg2_ids = self._add_decoding_toks(a2, a2_)
            trg2_txt = entry["a2"].tolist()[: self.la]
            trg2_txt = trg2_txt + [self.vocab.pad] * (self.la - len(trg2_txt))

            ## Append processed fields to list
            self.q_ids.append(np.array(q_ids))
            self.q_masks.append(np.array(q_masks))
            self.c_ids.append(np.array(c_ids))
            self.c_ids_ext.append(np.array(c_ids_ext))
            self.c_masks.append(np.array(c_masks))
            self.dec_inp1_ids.append(np.array(dec_inp1_ids))
            self.dec_inp2_ids.append(np.array(dec_inp2_ids))
            self.dec_trg1_ids.append(np.array(dec_trg1_ids))
            self.dec_trg2_ids.append(np.array(dec_trg2_ids))
            self.trg1_txt.append(" ".join(trg1_txt))
            self.trg2_txt.append(" ".join(trg2_txt))
            self.oovs.append(oovs)

    def _pad(self, s, max_len):
        pad_size = max_len - len(s)
        return s + [self.vocab.pad_id] * pad_size, [1] * len(s) + [0] * pad_size

    def _trunc(self, s, max_len):
        return s[:max_len]

    def _add_decoding_toks(self, a, a_):
        dec_input = [self.vocab.sos_id] + a_
        dec_target = a + [self.vocab.eos_id]
        # truncate inputs longer than max length

        dec_input = self._trunc(dec_input, self.la)
        dec_target = self._trunc(dec_target, self.la)
        dec_input, dec_input_masks = self._pad(dec_input, self.la)
        dec_target, _ = self._pad(dec_target, self.la)

        assert len(dec_input) == len(dec_target)

        return dec_input, dec_input_masks, dec_target

    def switch_answerability(self):
        self.exchange_rate = min((1, self.exchange_rate + 0.15))
