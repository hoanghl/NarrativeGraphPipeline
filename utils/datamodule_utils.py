import json

import numpy as np
import torch


class Vocab:
    def __init__(self, path_vocab) -> None:
        super().__init__()

        self.s2i = dict()
        self.i2s = dict()
        with open(path_vocab) as f:
            vocab = json.load(f)
        for i, tok in enumerate(vocab):
            self.s2i[tok] = i
            self.i2s[i] = tok

        self.sos = "[SOS]"
        self.eos = "[EOS]"
        self.pad = "[PAD]"
        self.unk = "[UNK]"
        self.sos_id = self.s2i[self.sos]
        self.eos_id = self.s2i[self.eos]
        self.pad_id = self.s2i[self.pad]
        self.unk_id = self.s2i[self.unk]

    def __len__(self):
        return len(self.s2i)

    def stoi(self, s):
        if isinstance(s, str):
            return [self.s2i[t] for t in s.split(" ").lower()]
        elif isinstance(s, list) or isinstance(s, np.ndarray):
            return [self.s2i[t.lower()] if t.lower() in self.s2i else self.unk_id for t in s]

        assert f"Type of 's' error: {type(s)}"

    def itos(self, indx: list):
        if torch.is_tensor(indx):
            indx = indx.tolist()
        return [self.i2s[i] for i in indx]

    def pred2s(self, indx, oov):
        if torch.is_tensor(indx):
            indx = indx.tolist()
        oov = {k: v for k, v in oov.items()}
        toks = []
        for i in indx:
            if i >= len(self):
                if i - len(self) in oov:
                    toks.append(oov[i - len(self)])
                else:
                    toks.append(self.unk)
            else:
                toks.append(self.i2s[i])

        return toks


def collate_fn(data):
    q_ids = torch.stack([torch.tensor(entry["q_ids"], dtype=torch.long) for entry in data])
    q_masks = torch.stack([torch.tensor(entry["q_masks"], dtype=torch.long) for entry in data])
    c_ids = torch.stack([torch.tensor(entry["c_ids"], dtype=torch.long) for entry in data])
    c_ids_ext = torch.stack([torch.tensor(entry["c_ids_ext"], dtype=torch.long) for entry in data])
    c_masks = torch.stack([torch.tensor(entry["c_masks"], dtype=torch.long) for entry in data])
    dec_inp1_ids = torch.stack(
        [torch.tensor(entry["dec_inp1_ids"], dtype=torch.long) for entry in data]
    )
    dec_inp2_ids = torch.stack(
        [torch.tensor(entry["dec_inp2_ids"], dtype=torch.long) for entry in data]
    )
    dec_trg1_ids = torch.stack(
        [torch.tensor(entry["dec_trg1_ids"], dtype=torch.long) for entry in data]
    )
    dec_trg2_ids = torch.stack(
        [torch.tensor(entry["dec_trg2_ids"], dtype=torch.long) for entry in data]
    )
    trg1_txt = [entry["trg1_txt"] for entry in data]
    trg2_txt = [entry["trg2_txt"] for entry in data]
    oovs = [entry["oovs"] for entry in data]

    return {
        "q_ids": q_ids,
        "q_masks": q_masks,
        "c_ids": c_ids,
        "c_ids_ext": c_ids_ext,
        "c_masks": c_masks,
        "dec_inp1_ids": dec_inp1_ids,
        "dec_inp2_ids": dec_inp2_ids,
        "dec_trg1_ids": dec_trg1_ids,
        "dec_trg2_ids": dec_trg2_ids,
        "trg1_txt": trg1_txt,
        "trg2_txt": trg2_txt,
        "oovs": oovs,
    }
