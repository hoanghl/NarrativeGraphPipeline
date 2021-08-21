import re

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from transformers.generation_utils import *


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim, ffn_dim=2048, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(nn.functional.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=True, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            scale = (k.size(2)) ** -0.5
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, float("-inf"))
        # softmax
        attention = self.softmax(attention)
        # dropout
        attention = self.dropout(attention)
        # dot product with V
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        # layer norm after multi-head attention
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        # residual connection
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # scaled dot product attention
        context, attention = self.dot_product_attention(query, key, value, attn_mask=attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        attention = attention.view(batch_size, num_heads, query.size(1), key.size(1))
        attention = attention.sum(dim=1) / num_heads

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(num_heads, model_dim, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, q, k, v, attn_mask=None):

        # attention
        context, attention = self.attention(q, k, v, attn_mask=attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim=2048, dropout=0.1):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, q, k, v, attn_mask=None):
        output = q
        attention = None
        for encoder in self.encoder_layers:
            output, attention = encoder(output, k, v, attn_mask=attn_mask)

        return output, attention


def get_scores(outputs):
    n_samples = 0
    bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
    for pair in outputs:
        bleu_1_, bleu_4_, meteor_, rouge_l_ = _get_scores(**pair)

        bleu_1 += bleu_1_
        bleu_4 += bleu_4_
        meteor += meteor_
        rouge_l += rouge_l_

        n_samples += 1

    return (
        bleu_1 / n_samples,
        bleu_4 / n_samples,
        meteor / n_samples,
        rouge_l / n_samples,
    )


def process_sent(sent):
    return re.sub(r"(\[PAD\]|\[CLS\]|\[SEP\]|\[UNK\]|\[MASK\])", "", sent).strip()


def _get_scores(ref: list, pred, eps=10e-8):
    """Calculate metrics BLEU-1, BLEU4, METEOR and ROUGE_L.

    ref = [
        "the transcript is a written version of each day",
        "version of each day"
    ]
    pred= "a written version of"

    Args:
        ref (list): list of reference strings
        pred (str): string generated by model

    Returns:
        tuple: tuple of 4 scores
    """

    pred = process_sent(pred)
    ref = list(map(process_sent, ref))

    if pred == "":
        return 0, 0, 0, 0

    # Calculate BLEU score
    ref_ = [x.split() for x in ref]
    pred_ = pred.split()

    bleu_1 = sentence_bleu(ref_, pred_, weights=(1, 0, 0, 0))
    bleu_4 = sentence_bleu(ref_, pred_, weights=(0.25, 0.25, 0.25, 0.25))

    # Calculate METEOR
    meteor = meteor_score(ref, pred)

    # Calculate ROUGE-L
    scores = np.array([Rouge().get_scores(ref_, pred, avg=True)["rouge-l"]["f"] for ref_ in ref])
    rouge_l = np.mean(scores)

    return (
        bleu_1 if bleu_1 > eps else 0,
        bleu_4 if bleu_4 > eps else 0,
        meteor if meteor > eps else 0,
        rouge_l if rouge_l > eps else 0,
    )
