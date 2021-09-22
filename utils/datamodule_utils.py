# Most of this file is copied from
# https://github.com/abisee/pointer-generator/blob/master/data.py
# https://github.com/atulkum/pointer_summarizer/blob/master/data_util/data.py

import json
import pickle

import numpy as np
import torch

PAD_TOKEN = "[PAD]"  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK_TOKEN = "[UNK]"  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = (
    "[SOS]"  # This has a vocab id, which is used at the start of every decoder input sequence
)
STOP_DECODING = (
    "[EOS]"  # This has a vocab id, which is used at the end of untruncated target sequences
)


class Vocab(object):
    """
    Vocabulary class for mapping between words and ids (integers)
    """

    def __init__(self):
        self._word_to_id = {}
        self._id_to_word = []
        self._count = 0

    @classmethod
    def from_json(cls, vocab_file):
        vocab = cls()
        with open(vocab_file, "r") as f:
            words = json.load(f)
        for i, tok in enumerate(words):
            vocab._word_to_id[tok] = i
            vocab._id_to_word.append(tok)

        vocab._count = len(vocab._id_to_word)
        vocab.specials = [PAD_TOKEN, UNK_TOKEN, START_DECODING, STOP_DECODING]
        return vocab

    def save(self, fpath):
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(self._word_to_id, f, ensure_ascii=False, indent=4)
        print(f"vocab file saved as {fpath}")

    def __len__(self):
        """Returns size of the vocabulary."""
        return self._count

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        unk_id = self.unk()
        return self._word_to_id.get(word, unk_id)

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError(f"Id not found in vocab: {word_id}")
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary."""
        return self._count

    def pad(self):
        """Helper to get index of pad symbol"""
        return self._word_to_id[PAD_TOKEN]

    def unk(self):
        """Helper to get index of unk symbol"""
        return self._word_to_id[UNK_TOKEN]

    def start(self):
        return self._word_to_id[START_DECODING]

    def stop(self):
        return self._word_to_id[STOP_DECODING]

    def extend(self, oovs):
        extended_vocab = self._id_to_word + list(oovs)
        return extended_vocab

    def tokens2ids(self, tokens):
        ids = [self.word2id(t) for t in tokens]
        return ids

    def source2ids_ext(self, src_tokens):
        """Maps source tokens to ids if in vocab, extended vocab ids if oov.

        Args:
            src_tokens: list of source text tokens

        Returns:
            ids: list of source text token ids
            oovs: list of oovs in source text
        """
        ids = []
        oovs = []
        for t in src_tokens:
            t_id = self.word2id(t)
            unk_id = self.word2id(UNK_TOKEN)
            if t_id == unk_id:
                if t not in oovs:
                    oovs.append(t)
                ids.append(self.size() + oovs.index(t))
            else:
                ids.append(t_id)
        return ids, oovs

    def target2ids_ext(self, tgt_tokens, oovs):
        """Maps target text to ids, using extended vocab (vocab + oovs).

        Args:
            tgt_tokens: list of target text tokens
            oovs: list of oovs from source text (copy mechanism)

        Returns:
            ids: list of target text token ids
        """
        ids = []
        for t in tgt_tokens:
            t_id = self.word2id(t)
            unk_id = self.word2id(UNK_TOKEN)
            if t_id == unk_id:
                if t in oovs:
                    ids.append(self.size() + oovs.index(t))
                else:
                    ids.append(unk_id)
            else:
                ids.append(t_id)
        return ids

    def outputids2words(self, ids, src_oovs):
        """Maps output ids to words

        Args:
            ids: list of ids
            src_oovs: list of oov words

        Returns:
            words: list of words mapped from ids

        """
        words = []
        extended_vocab = self.extend(src_oovs)
        for i in ids:
            try:
                w = self.id2word(i)  # might be oov
            except ValueError as e:
                assert (
                    src_oovs is not None
                ), "Error: model produced a word ID that isn't in the vocabulary."
                try:
                    w = extended_vocab[i]
                except IndexError as e:
                    raise ValueError(
                        f"Error: model produced word ID {i} \
                                       but this example only has {len(src_oovs)} article OOVs"
                    )
            words.append(w)
        return words


class Batch:
    def __init__(self, data, vocab, max_decode):
        q, q_len, c, c_len, a, a_len = list(zip(*data))
        self.vocab = vocab
        self.pad_id = self.vocab.pad()
        self.max_decode = max_decode

        self.q_input, self.q_len, self.q_pad_mask = None, None, None
        # Encoder info
        self.c_input, self.c_len, self.c_pad_mask = None, None, None
        # Additional info for pointer-generator network
        self.c_input_ext, self.max_oov_len, self.src_oovs = None, None, None
        # Decoder info
        self.dec_input, self.dec_target, self.dec_len, self.dec_pad_mask = None, None, None, None

        # Build batch inputs
        self.init_question_seq(q, q_len)
        self.init_context_seq(c, c_len)
        self.init_answer_seq(a, a_len)

        # Save original strings
        self.q_text = c
        self.c_text = c
        self.a_text = a

    def init_question_seq(self, q, q_len):
        q_ids = [self.vocab.tokens2ids(s) for s in q]

        self.q_input = collate_tokens(values=q_ids, pad_idx=self.pad_id)
        self.q_len = torch.LongTensor(q_len)
        self.q_pad_mask = self.q_input == self.pad_id

    def init_context_seq(self, src, src_len):
        src_ids = [self.vocab.tokens2ids(s) for s in src]

        self.c_input = collate_tokens(values=src_ids, pad_idx=self.pad_id)
        self.c_len = torch.LongTensor(src_len)
        self.c_pad_mask = self.c_input == self.pad_id

        # Save additional info for pointer-generator
        # Determine max number of source text OOVs in this batch
        src_ids_ext, oovs = zip(*[self.vocab.source2ids_ext(s) for s in src])
        # Store the version of the encoder batch that uses article OOV ids
        self.c_input_ext = collate_tokens(values=src_ids_ext, pad_idx=self.pad_id)
        self.max_oov_len = max([len(oov) for oov in oovs])
        # Store source text OOVs themselves
        self.src_oovs = oovs

    def init_answer_seq(self, tgt, tgt_len):
        tgt_ids = [self.vocab.tokens2ids(t) for t in tgt]
        tgt_ids_ext = [self.vocab.target2ids_ext(t, oov) for t, oov in zip(tgt, self.src_oovs)]

        # create decoder inputs
        dec_input, _ = zip(*[self.get_decoder_input_target(t, self.max_decode) for t in tgt_ids])

        self.dec_input = collate_tokens(
            values=dec_input, pad_idx=self.pad_id, pad_to_length=self.max_decode
        )

        # create decoder targets using extended vocab
        _, dec_target = zip(
            *[self.get_decoder_input_target(t, self.max_decode) for t in tgt_ids_ext]
        )

        self.dec_target = collate_tokens(
            values=dec_target, pad_idx=self.pad_id, pad_to_length=self.max_decode
        )

        self.dec_len = torch.LongTensor(tgt_len)
        self.dec_pad_mask = self.dec_input == self.pad_id

    def get_decoder_input_target(self, tgt, max_len):
        dec_input = [self.vocab.start()] + tgt
        dec_target = tgt + [self.vocab.stop()]
        # truncate inputs longer than max length
        if len(dec_input) > max_len:
            dec_input = dec_input[:max_len]
            dec_target = dec_target[:max_len]
        assert len(dec_input) == len(dec_target)
        return dec_input, dec_target

    def __len__(self):
        return self.c_input.size(0)

    def __str__(self):
        batch_info = {
            "q_text": self.q_text,
            "c_text": self.c_text,
            "a_text": self.a_text,
            "q_input": self.q_input,  # [B x L]
            "q_len": self.q_len,  # [B]
            "q_pad_mask": self.q_pad_mask,  # [B x L]
            "c_input": self.c_input,  # [B x L]
            "c_input_ext": self.c_input_ext,  # [B x L]
            "c_len": self.c_len,  # [B]
            "c_pad_mask": self.c_pad_mask,  # [B x L]
            "src_oovs": self.src_oovs,  # list of length B
            "max_oov_len": self.max_oov_len,  # single int value
            "dec_input": self.dec_input,  # [B x T]
            "dec_target": self.dec_target,  # [B x T]
            "dec_len": self.dec_len,  # [B]
            "dec_pad_mask": self.dec_pad_mask,  # [B x T]
        }
        return str(batch_info)

    def to(self, device):
        self.q_input = self.q_input.to(device)
        self.q_len = self.q_len.to(device)
        self.q_pad_mask = self.q_pad_mask.to(device)

        self.c_input = self.c_input.to(device)
        self.c_input_ext = self.c_input_ext.to(device)
        self.c_len = self.c_len.to(device)
        self.c_pad_mask = self.c_pad_mask.to(device)

        self.dec_input = self.dec_input.to(device)
        self.dec_target = self.dec_target.to(device)
        self.dec_len = self.dec_len.to(device)
        self.dec_pad_mask = self.dec_pad_mask.to(device)
        return self


def load_data(path):
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    q, cE, a1 = [], [], []

    for entry in dataset:
        q.append(entry["q"].tolist())
        cE.append(np.concatenate(entry["cE"]).tolist())
        a1.append(entry["a1"].tolist())

    return q, cE, a1


def collate_tokens(values, pad_idx, left_pad=False, pad_to_length=None):
    # Simplified version of `collate_tokens` from fairseq.data.data_utils
    """Convert a list of 1d tensors into a padded 2d tensor."""
    values = list(map(torch.LongTensor, values))
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res
