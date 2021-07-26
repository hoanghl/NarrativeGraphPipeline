from random import sample
import glob

from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np


from src.utils.utils import ParallelHelper


class NarrativeDataset(Dataset):
    def __init__(
        self,
        split: str,
        path_data: str,
        path_bert: str,
        size_dataset: int,
        l_q,
        l_c,
        l_a,
        n_paras,
        num_worker,
    ):

        self.split = split
        self.l_q = l_q
        self.l_c = l_c
        self.l_a = l_a
        self.n_paras = n_paras
        self.num_workers = num_worker
        self.size_dataset = size_dataset

        path_data = path_data.replace("[SPLIT]", split).replace("[SHARD]", "*")
        self.paths = sorted(glob.glob(path_data))

        self.tokenizer = BertTokenizer.from_pretrained(path_bert)

        self.curent_ith_file = -1

        self.ques_ids = None
        self.ques_mask = None
        self.ans1_ids = None
        self.ans2_ids = None
        self.ans1_mask = None
        self.context_ids = None
        self.context_mask = None

        self.exchange_rate = 0

    def __len__(self) -> int:
        return self.size_dataset

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        size_shard = self.size_dataset // 8

        ith_file = indx // size_shard
        indx = indx % size_shard

        if ith_file == 8:
            ith_file -= 1
            indx = indx + size_shard

        # Check nth file and reload dataset if needed
        if ith_file != self.curent_ith_file:
            self.curent_ith_file = ith_file

            # Reload dataset
            # gc.collect()
            self.read_datasetfile(self.paths[self.curent_ith_file])

        return {
            # 'docId'         : self.docId[indx],
            # 'ques_plain'    : self.ques_plain[indx],
            "ques_ids": self.ques_ids[indx],
            "ques_mask": self.ques_mask[indx],
            "ans1_ids": self.ans1_ids[indx],
            "ans2_ids": self.ans2_ids[indx],
            "ans1_mask": self.ans1_mask[indx],
            "context_ids": self.context_ids[indx],
            "context_mask": self.context_mask[indx],
        }

    def _get_context(self, En, Hn):
        n_samples = min((len(En), self.n_paras))
        if self.split == "train":
            selects_Hn = int(n_samples * self.exchange_rate)
            selects_En = n_samples - selects_Hn

            return sample(En, selects_En) + sample(Hn, selects_Hn)

        return sample(Hn, n_samples)

    def f_process_file_multi(self, entries, queue):
        for entry in entries.itertuples():
            queue.put(self.f_process_file_single(entry))

    def f_process_file_single(self, entry):
        ###########################
        # Process question
        ###########################
        encoded = self.tokenizer(
            entry.question,
            padding="max_length",
            max_length=self.l_q,
            truncation=True,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ques_ids = encoded["input_ids"][0]
        ques_mask = encoded["attention_mask"][0]

        ###########################
        # Process answers
        ###########################
        ans1, ans2 = entry.answers

        # This trick ensures training process occurs in longer answer
        if len(" ".split(ans1)) < len(" ".split(ans2)):
            ans1, ans2 = ans2, ans1

        encoded = self.tokenizer(
            ans1,
            padding="max_length",
            truncation=True,
            max_length=self.l_a,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ans1_ids = encoded["input_ids"][0]
        ans1_mask = encoded["attention_mask"][0]

        encoded = self.tokenizer(
            ans2,
            padding="max_length",
            truncation=True,
            max_length=self.l_a,
            return_tensors="np",
            return_token_type_ids=False,
        )
        ans2_ids = encoded["input_ids"][0]

        ###########################
        # Process context
        ###########################
        En = entry.En.tolist()
        Hn = entry.Hn.tolist()

        contx = self._get_context(En, Hn)

        # Process context
        context_ids = np.zeros((self.n_paras, self.l_c), dtype=np.int)
        context_mask = np.zeros((self.n_paras, self.l_c), dtype=np.int)
        for ith, para in enumerate(contx):
            encoded = self.tokenizer(
                para,
                padding="max_length",
                truncation=True,
                max_length=self.l_c,
                return_tensors="np",
                return_token_type_ids=False,
            )
            context_ids[ith] = encoded["input_ids"]
            context_mask[ith] = encoded["attention_mask"]

        return {
            "ques_ids": ques_ids,
            "ques_mask": ques_mask,
            "ans1_ids": ans1_ids,
            "ans2_ids": ans2_ids,
            "ans1_mask": ans1_mask,
            "context_ids": context_ids,
            "context_mask": context_mask,
        }

    def read_datasetfile(self, path_file):
        df = pd.read_parquet(path_file)

        # self.docId          = []
        # self.ques_plain     = []
        self.ques_ids = []
        self.ques_mask = []
        self.ans1_ids = []
        self.ans2_ids = []
        self.ans1_mask = []
        self.context_ids = []
        self.context_mask = []

        # gc.collect()

        ######################
        # Fill self.ques_ids, self.ans1_ids,  self.ans2_ids,
        # answers' mask and index
        ######################
        if self.num_workers > 1:
            entries = ParallelHelper(
                self.f_process_file_multi,
                df,
                lambda dat, l, h: dat.iloc[l:h],
                self.num_workers,
            ).launch()
        else:
            entries = list(map(self.f_process_file_single, df.itertuples()))

        for entry in entries:
            # self.docId.append(entry['docId'])
            # self.ques_plain.append(entry['ques_plain'])
            self.ques_ids.append(entry["ques_ids"])
            self.ques_mask.append(entry["ques_mask"])
            self.ans1_ids.append(entry["ans1_ids"])
            self.ans2_ids.append(entry["ans2_ids"])
            self.ans1_mask.append(entry["ans1_mask"])
            self.context_ids.append(entry["context_ids"])
            self.context_mask.append(entry["context_mask"])

    def switch_answerability(self):
        if self.exchange_rate == 1:
            self.exchange_rate = 0
        else:
            self.exchange_rate += 0.25
