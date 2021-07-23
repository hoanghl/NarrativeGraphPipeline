"""This file contains code reading raw data and do some preprocessing"""
import re, json, os, logging, gc

from transformers import BertTokenizerFast
from rank_bm25 import BM25Okapi
from omegaconf import OmegaConf
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
import unidecode
import dotenv

import utils.utils as utils


logging.getLogger("spacy").setLevel(logging.ERROR)
dotenv.load_dotenv(override=True)
log = utils.get_logger()


class ContextProcessor:
    def __init__(
        self,
        l_c_contx_process,
        path_raw,
        path_processed_contx,
        path_data,
        n_workers,
    ):
        self.nlp_spacy = spacy.load("en_core_web_sm")
        self.nlp_spacy.add_pipe("sentencizer")

        self.path_raw = path_raw
        self.l_c_contx_process = l_c_contx_process
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.n_workers = n_workers

    def read_contx(self, id_):
        with open(
            f"{self.path_raw}/texts/{id_}.content", "r", encoding="iso-8859-1"
        ) as d_file:
            return d_file.read()

    def clean_end(self, text):
        text = text.split(" ")
        return " ".join(text[:-2])

    def clean_context_movie(self, context) -> str:
        context = context.lower().strip()

        context = re.sub(r"(style\=\".*\"\>|class\=.*\>)", "", context)

        context = re.sub(r" \n ", "\n", context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r"(?<=\w|,)\n(?=\w| )", " ", context)
        context = re.sub(r"\n{2,}", "\n", context)

        return context

    def clean_context_gutenberg(self, context) -> str:
        context = context.lower().strip()

        context = re.sub(r"(style\=\".*\"\>|class\=.*\>)", "", context)

        context = re.sub(r"(?<=\w|,|;|-)\n(?=\w| )", " ", context)
        context = re.sub(r"( {2,}|\t)", " ", context)
        # context = re.sub(r' \n ', '\n', context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r"\n{2,}", "\n", context)

        return context

    def export_para(self, toks):
        para_ = []
        for i in range(0, len(toks), self.l_c_contx_process):
            tmp = re.sub(
                r"( |\t){2,}", " ", " ".join(toks[i : i + self.l_c_contx_process])
            ).strip()
            para_.append(tmp)

        return np.array(para_)

        # return re.sub(r'( {2,}|\t)', ' ', ' '.join(toks)).strip()

    def extract_html(self, context):
        soup = BeautifulSoup(context, "html.parser")
        context = unidecode.unidecode(soup.text)
        return context

    def extract_start_end(self, context, start, end):
        end = self.clean_end(end)

        start = context.find(start)
        end = context.rfind(end)
        if start == -1:
            start = 0
        if end == -1:
            end = len(context)
        if start >= end:
            start, end = 0, len(context)

        context = context[start:end]

        return context

    def f_process_contx(self, entries, queue):
        for entry in entries.itertuples():
            docId = entry.document_id

            path = self.path_processed_contx.replace("[ID]", docId)
            if os.path.exists(path):
                queue.put(1)
                continue

            context = self.read_contx(docId)
            kind = entry.kind
            start = entry.story_start.lower()
            end = entry.story_end.lower()

            ## Extract text from HTML
            context = self.extract_html(context)

            ## Clean context and split into paras
            if kind == "movie":
                context = self.clean_context_movie(context)
            else:
                context = self.clean_context_gutenberg(context)

            ## Use field 'start' and 'end' provided
            sentences = self.extract_start_end(context, start, end).split("\n")

            tokens, paras = np.array([]), np.array([])
            for sent in sentences:
                ## Tokenize
                tokens_ = [tok.text for tok in self.nlp_spacy(sent)]
                tokens = np.concatenate((tokens, tokens_))
            l_c = self.l_c_contx_process
            for i in range(0, len(tokens), l_c):
                para = " ".join(tokens[i : i + l_c])
                para = re.sub(r"( |\t){2,}", " ", para).strip()

                paras = np.concatenate((paras, [para]))

            with open(path, "w+") as contx_file:
                json.dump(paras.tolist(), contx_file, indent=2, ensure_ascii=False)

            queue.put(1)

    def trigger_process_contx(self):
        log.info(" = Process context.")

        documents = pd.read_csv(
            f"{self.path_raw}/documents.csv", header=0, index_col=None
        )

        for split in ["train", "test", "valid"]:
            log.info(f" = Process context of split: {split}")

            path_dir = os.path.dirname(self.path_processed_contx)
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir, exist_ok=True)

            utils.ParallelHelper(
                self.f_process_contx,
                documents[documents["set"] == split],
                lambda d, l, h: d.iloc[l:h],
                self.n_workers,
                show_bar=True,
            ).launch()


class DataProcessor:
    def __init__(
        self,
        l_q,
        l_c,
        l_a,
        n_c,
        path_raw,
        path_processed_contx,
        path_data,
        path_bert,
        n_workers,
        n_shards,
    ):

        self.l_q = l_q
        self.l_c = l_c
        self.l_a = l_a
        self.n_c = n_c
        self.path_raw = path_raw
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.n_workers = n_workers
        self.n_shards = n_shards

        self.nlp_spacy = spacy.load("en_core_web_sm")
        self.tokenizer = BertTokenizerFast.from_pretrained(path_bert)

    def process_ques_ans(self, text):
        """Process question/answers

        Args:
            text (str): question or answers

        Returns:
            str: processed result
        """
        tok = [tok.text for tok in self.nlp_spacy(text) if not tok.is_punct]

        return " ".join(tok)

    def read_processed_contx(self, id_):
        path = self.path_processed_contx.replace("[ID]", id_)
        assert os.path.isfile(path), f"Context with id {id_} not found."
        with open(path, "r") as d_file:
            return json.load(d_file)

    def f_process_entry_multi(self, entries, queue):
        """This function is used to run in parallel tailored for utils.ParallelHelper."""

        for entry in entries.itertuples():
            queue.put(self.f_process_entry(entry))

    def f_process_entry(self, entry):
        c = np.array(self.read_processed_contx(entry.document_id))

        # Find topK (= n_c) BM25-score paras
        q = self.process_ques_ans(entry.question)
        a1 = self.process_ques_ans(entry.answer1)
        a2 = self.process_ques_ans(entry.answer2)
        if len(a1.split(" ")) < len(a2.split(" ")):
            a1, a2 = a2, a1
        a = a1 + " " + a2

        tokenized_c = [para.split(" ") for para in c]
        bm25 = BM25Okapi(tokenized_c)

        scores_q = bm25.get_scores(q)
        c_H = c[np.where(scores_q.shape[0] - scores_q.argsort() <= self.n_c)].tolist()
        scores_a = bm25.get_scores(a)
        c_E = c[np.where(scores_a.shape[0] - scores_a.argsort() <= self.n_c)].tolist()

        c_E = self.tokenizer(
            c_E,
            padding="max_length",
            truncation=True,
            max_length=self.l_c,
            return_tensors="np",
            return_token_type_ids=False,
        )
        c_H = self.tokenizer(
            c_H,
            padding="max_length",
            truncation=True,
            max_length=self.l_c,
            return_tensors="np",
            return_token_type_ids=False,
        )

        # Tokenize answers
        a1 = self.tokenizer(
            a1,
            padding="max_length",
            truncation=True,
            max_length=self.l_a,
            return_tensors="np",
            return_token_type_ids=False,
        )

        a2 = self.tokenizer(
            a2,
            padding="max_length",
            truncation=True,
            max_length=self.l_a,
            return_tensors="np",
            return_token_type_ids=False,
        )

        # Tokenize question
        q = self.tokenizer(
            q,
            padding="max_length",
            truncation=True,
            max_length=self.l_q,
            return_tensors="np",
            return_token_type_ids=False,
        )

        return {
            "q_ids": q["input_ids"].flatten(),
            "q_masks": q["attention_mask"].flatten(),
            "c_E_ids": c_E["input_ids"].flatten(),
            "c_E_masks": c_E["attention_mask"].flatten(),
            "c_H_ids": c_H["input_ids"].flatten(),
            "c_H_masks": c_H["attention_mask"].flatten(),
            "a1_ids": a1["input_ids"].flatten(),
            "a1_masks": a1["attention_mask"].flatten(),
            "a2_ids": a2["input_ids"].flatten(),
        }

    def trigger_create_training_data(self):
        log.info("Start creating data for training")

        documents = pd.read_csv(f"{self.path_raw}/qaps.csv", header=0, index_col=None)

        for split in ["train", "test", "valid"]:
            documents_ = documents[documents["set"] == split]
            for shard in range(self.n_shards):
                ### Need to check whether this shard has already been processed
                path = self.path_data.replace("[SPLIT]", split).replace(
                    "[SHARD]", f"{shard:02d}"
                )
                if os.path.exists(path):
                    continue

                ## Make dir to contain processed files
                os.makedirs(os.path.dirname(path), exist_ok=True)

                ## Start processing (multi/single processing)
                start_ = len(documents_) // self.n_shards * shard
                end_ = (
                    start_ + len(documents_) // self.n_shards
                    if shard < self.n_shards - 1
                    else len(documents_)
                )

                if self.n_workers == 1:
                    list_documents = list(
                        map(
                            self.f_process_entry,
                            tqdm(
                                documents_.iloc[start_:end_].itertuples(),
                                total=end_ - start_,
                                desc=f"Split {split} - shard {shard:02d}",
                            ),
                        )
                    )
                else:

                    list_documents = utils.ParallelHelper(
                        self.f_process_entry_multi,
                        documents_.iloc[start_:end_],
                        lambda d, l, h: d.iloc[l:h],
                        self.n_workers,
                        desc=f"Split {split} - shard {shard:02d}",
                        show_bar=True,
                    ).launch()

                ## Save processed things to Parquet file
                if len(list_documents) > 0:
                    df = pd.DataFrame(list_documents)

                    df.to_parquet(path)

                    df = None
                    gc.collect()


class Preprocess:
    def __init__(
        self,
        l_q,
        l_c,
        l_a,
        n_c,
        l_c_contx_process,
        n_workers,
        n_shards,
        path_raw,
        path_bert,
        path_processed_contx,
        path_data,
    ):

        ######################
        # Define processors
        ######################
        self.contx_processor = ContextProcessor(
            l_c_contx_process=l_c_contx_process,
            path_raw=path_raw,
            path_processed_contx=path_processed_contx,
            path_data=path_data,
            n_workers=n_workers,
        )
        self.data_processor = DataProcessor(
            l_q=l_q,
            l_c=l_c,
            l_a=l_a,
            n_c=n_c,
            path_raw=path_raw,
            path_processed_contx=path_processed_contx,
            path_data=path_data,
            path_bert=path_bert,
            n_workers=n_workers,
            n_shards=n_shards,
        )

    def preprocess(self):
        # self.contx_processor.trigger_process_contx()
        self.data_processor.trigger_create_training_data()


if __name__ == "__main__":
    # Read config from 'configs.yaml'
    config = OmegaConf.load("configs.yaml")
    config.PATH.utils = os.environ.get("NARRATIVE_UTILS")

    OmegaConf.resolve(config)

    Preprocess(
        l_q=config.l_q,
        l_c=config.l_c,
        l_a=config.l_a,
        n_c=config.n_c,
        l_c_contx_process=config.l_c_contx_process,
        n_workers=config.n_workers,
        n_shards=config.n_shards,
        path_raw=config.PATH.raw,
        path_bert=config.PATH.bert,
        path_processed_contx=config.PATH.processed_contx,
        path_data=config.PATH.data,
    ).preprocess()
