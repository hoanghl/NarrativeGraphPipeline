"""This file contains code reading raw data and do some preprocessing"""
from collections import defaultdict
from glob import glob
import re, json, os, logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
import unidecode

from src.utils.utils import ParallelHelper


log = logging.getLogger("spacy")
log.setLevel(logging.ERROR)


class ContextProcessor:
    def __init__(
        self,
        nlp_spacy: spacy.Language,
        l_c_processing,
        path_raw_data,
        path_processed_contx,
        path_data,
        num_workers,
    ):
        self.nlp_spacy = nlp_spacy

        self.path_raw_data = path_raw_data
        self.l_c_processing = l_c_processing
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.num_workers = num_workers

    def read_contx(self, id_):
        with open(
            f"{self.path_raw_data}/texts/{id_}.content", "r", encoding="iso-8859-1"
        ) as d_file:
            return d_file.read()

    def clean_end(self, text: str):
        text = text.split(" ")
        return " ".join(text[:-2])

    def clean_context_movie(self, context: str) -> str:
        context = context.lower().strip()

        context = re.sub(r"(style\=\".*\"\>|class\=.*\>)", "", context)

        context = re.sub(r" \n ", "\n", context)
        # context = re.sub(r'\n ', ' ', context)
        context = re.sub(r"(?<=\w|,)\n(?=\w| )", " ", context)
        context = re.sub(r"\n{2,}", "\n", context)

        return context

    def clean_context_gutenberg(self, context: str) -> str:
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
        for i in range(0, len(toks), self.l_c_processing):
            tmp = re.sub(
                r"( |\t){2,}", " ", " ".join(toks[i : i + self.l_c_processing])
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
            l_c = self.l_c_processing
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
            f"{self.path_raw_data}/documents.csv", header=0, index_col=None
        )

        for split in ["train", "test", "valid"]:
            log.info(f" = Process context of split: {split}")

            path_dir = os.path.dirname(self.path_processed_contx)
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir, exist_ok=True)

            ParallelHelper(
                self.f_process_contx,
                documents[documents["set"] == split],
                lambda d, l, h: d.iloc[l:h],
                self.num_workers,
                show_bar=True,
            ).launch()


class EntryProcessor:
    def __init__(
        self,
        nlp_spacy: spacy.Language,
        n_paras,
        path_raw_data,
        path_processed_contx,
        path_data,
        num_workers,
    ):
        self.nlp_spacy = nlp_spacy

        self.n_paras = n_paras
        self.path_raw_data = path_raw_data
        self.path_processed_contx = path_processed_contx
        self.path_data = path_data
        self.num_workers = num_workers

    def process_ques_ans(self, text):
        """Process question/answers

        Args:
            text (str): question or answers

        Returns:
            str: processed result
        """
        tok = [tok.text for tok in self.nlp_spacy(text) if not tok.is_punct]

        return " ".join(tok)

    def find_golden(self, query, wm: list) -> set:

        ## Calculate score of query for each para
        query_ = np.expand_dims(wm[query], 0)
        wm_ = wm[:-2]

        scores = cosine_similarity(query_, wm_).squeeze(0)

        # Sort scores in descending and get corresponding indices
        score_indx = [(i, s_) for i, s_ in enumerate(scores.tolist())]
        score_indx.sort(key=lambda x: x[1], reverse=True)
        indices = [indx for indx, _ in score_indx]

        return indices[: self.n_paras]

    def read_processed_contx(self, id_):
        path = self.path_processed_contx.replace("[ID]", id_)
        assert os.path.isfile(path), f"Context with id {id_} not found."
        with open(path, "r") as d_file:
            return json.load(d_file)

    def f_process_entry(self, entry):
        """This function is used to run in parallel tailored for list mapping."""

        paras = self.read_processed_contx(entry.document_id)

        #########################
        ## Preprocess question and answer
        #########################
        ques = self.process_ques_ans(entry.question)
        ans1 = self.process_ques_ans(entry.answer1)
        ans2 = self.process_ques_ans(entry.answer2)
        ans = ans1 + " " + ans2

        #########################
        ## TfIdf vectorize
        #########################
        tfidfvectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words="english",
            ngram_range=(1, 3),
            max_features=500000,
        )

        wm = tfidfvectorizer.fit_transform(paras + [ques, ans]).toarray()

        ques, ans = len(wm) - 2, len(wm) - 1

        #########################
        ## Find golden paragraphs from
        ## question and answers
        #########################
        golden_paras_ques = self.find_golden(ques, wm)
        golden_paras_answ = self.find_golden(ans, wm)

        return {
            "doc_id": entry.document_id,
            "question": entry.question_tokenized.lower(),
            "answers": [
                entry.answer1_tokenized.lower(),
                entry.answer2_tokenized.lower(),
            ],
            "En": [paras[i] for i in golden_paras_answ],
            "Hn": [paras[i] for i in golden_paras_ques],
        }

    def trigger_process_entries(self):
        """Start processing pairs of question - context - answer"""
        documents = pd.read_csv(
            f"{self.path_raw_data}/qaps.csv", header=0, index_col=None
        )

        for split in ["train", "test", "valid"]:
            documents_ = documents[documents["set"] == split]
            for shard in range(8):
                ### Need to check whether this shard has already been processed
                path = self.path_data.replace("[SPLIT]", split).replace(
                    "[SHARD]", str(shard)
                )
                if os.path.exists(path):
                    continue

                ## Make dir to contain processed files
                os.makedirs(os.path.dirname(path), exist_ok=True)

                ## Start processing (multi/single processing)
                start_ = len(documents_) // 8 * shard
                end_ = start_ + len(documents_) // 8 if shard < 7 else len(documents_)

                if self.num_workers == 1:
                    list_documents = list(
                        map(
                            self.f_process_entry,
                            tqdm(
                                documents_.iloc[start_:end_].itertuples(),
                                total=end_ - start_,
                            ),
                        )
                    )
                else:

                    def f_process_entry_multi(entries, queue):
                        """This function is used to run in parallel tailored for ParallelHelper."""

                        for entry in entries.itertuples():
                            queue.put(self.f_process_entry(entry))

                    list_documents = ParallelHelper(
                        f_process_entry_multi,
                        documents_.iloc[start_:end_],
                        lambda d, l, h: d.iloc[l:h],
                        self.num_workers,
                        desc=f"Split {split} - shard {shard}",
                        show_bar=True,
                    ).launch()

                ## Save processed things to Parquet file
                if len(list_documents) > 0:
                    df = pd.DataFrame(list_documents)

                    df.to_parquet(path)


# class VocabBuilder:
#     def __init__(
#         self,
#         nlp_spacy: spacy.Language,
#         path_processed_contx: str,
#         path_raw_data: str,
#         path_vocab_gen: str,
#         path_vocab: str,
#     ) -> None:
#         self.nlp_spacy = nlp_spacy
#         self.path_processed_contx = path_processed_contx
#         self.path_raw_data = path_raw_data
#         self.path_vocab_gen = path_vocab_gen
#         self.path_vocab = path_vocab

#         self.nlp_spacy.disable_pipes(["ner", "parser", "tagger"])
#         self.nlp_spacy.max_length = 2500000

#     def build_vocab(self):
#         """Build vocab for generation and extended vocab"""

#         vocab_gen = defaultdict(int)
#         vocab_ex = set()

#         #########################
#         ## Read processed contexts and add words
#         #########################
#         paths_contx = glob(self.path_processed_contx.replace("[ID]", "*"))
#         assert len(paths_contx) > 0

#         for path in paths_contx:
#             bag_words = set()

#             with open(path, "r") as d_file:
#                 context = json.load(d_file)

#             for para in context:
#                 for tok in self.nlp_spacy(para):
#                     if not tok.like_url:
#                         vocab_ex.add(tok.text)

#                         if tok not in bag_words:
#                             vocab_gen[tok.text] += 1
#                             bag_words.add(tok.text)

#         ## Filter words appearing in at least 10 contexts
#         vocab_gen = sorted(vocab_gen.items(), key=lambda item: item[1], reverse=True)
#         vocab_gen = set(word for word, occurence in vocab_gen if occurence >= 10)

#         #########################
#         ## Read question and answers and add words
#         #########################
#         qaps = pd.read_csv(f"{self.path_raw_data}/qaps.csv", header=0, index_col=None)

#         for entry in qaps.itertuples():
#             tok_ques = entry.question_tokenized.lower().split(" ")
#             tok_ans1 = entry.answer1_tokenized.lower().split(" ")
#             tok_ans2 = entry.answer2_tokenized.lower().split(" ")

#             vocab_ex.update(tok_ques)
#             vocab_ex.update(tok_ans1)
#             vocab_ex.update(tok_ans2)

#         #########################
#         ## Write vocabs
#         #########################
#         # Write gen vocab
#         with open(self.path_vocab_gen, "w+") as f:
#             for word in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]:
#                 f.write(word + "\n")

#             for word in vocab_gen:
#                 f.write(word + "\n")

#         # Write extended vocab
#         with open(self.path_vocab, "w+") as f:
#             for word in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]:
#                 f.write(word + "\n")

#             for word in vocab_gen:
#                 f.write(word + "\n")

#             for word in vocab_ex - vocab_gen:
#                 f.write(word + "\n")


class Preprocess:
    def __init__(
        self,
        num_workers,
        l_c_processing,
        n_paras,
        path_raw_data,
        path_processed_contx,
        path_data,
        path_vocab_gen,
        path_vocab,
    ):

        nlp_spacy = spacy.load("en_core_web_sm")
        nlp_spacy.add_pipe("sentencizer")

        ######################
        # Define processors
        ######################
        self.contx_processor = ContextProcessor(
            nlp_spacy=nlp_spacy,
            l_c_processing=l_c_processing,
            path_raw_data=path_raw_data,
            path_processed_contx=path_processed_contx,
            path_data=path_data,
            num_workers=num_workers,
        )
        self.entry_processor = EntryProcessor(
            nlp_spacy=nlp_spacy,
            n_paras=n_paras,
            path_raw_data=path_raw_data,
            path_processed_contx=path_processed_contx,
            path_data=path_data,
            num_workers=num_workers,
        )
        # self.vocab_builder = VocabBuilder(
        #     nlp_spacy=nlp_spacy,
        #     path_processed_contx=path_processed_contx,
        #     path_raw_data=path_raw_data,
        #     path_vocab_gen=path_vocab_gen,
        #     path_vocab=path_vocab,
        # )

    def preprocess(self):
        self.contx_processor.trigger_process_contx()
        self.entry_processor.trigger_process_entries()

        # self.vocab_builder.build_vocab()


if __name__ == "__main__":
    Preprocess(
        num_workers=1,
        l_c_processing=150,
        n_paras=5,
        path_raw_data="/root/NarrativeQA",
        path_processed_contx="/root/data/proc_contx/[ID].json",
        path_data="/root/data/[SPLIT]/data_[SHARD].parquet",
        path_vocab_gen="/root/data/vocab_gen.txt",
        path_vocab="/root/data/vocab.txt",
    ).preprocess()

    # Preprocess(
    #     num_workers=8,
    #     l_c_processing=150,
    #     n_paras=5,
    #     path_raw_data="/root/NarrativeQA",
    #     path_processed_contx="data/proc_contx/[ID].json",
    #     path_data="data/[SPLIT]/data_[SHARD].parquet",
    # ).preprocess()
