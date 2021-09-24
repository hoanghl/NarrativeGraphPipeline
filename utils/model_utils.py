import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from transformers.generation_utils import *


def ipot(a1, a2, beta=2, max_iter=100, L=1):
    """Calculate loss based on OT."""

    b, la, d_hid = a1.size()
    n = b * la

    # a1: [b, la, d_hid]
    # a2: [b, la, d_hid]

    a1, a2 = a1.reshape(-1, d_hid), a2.reshape(-1, d_hid)
    # [n, d_hid]

    # Calculate matrix C
    a1_norm = a1 / a1.norm(dim=1)[:, None]
    a2_norm = a2 / a2.norm(dim=1)[:, None]
    C = a1_norm @ a2_norm.transpose(0, 1)
    # [n, n]

    sigma = torch.ones((n, 1), device=a1.device) / n

    T = torch.ones((n, n), device=a1.device) / n ** 2
    # [n, n]
    A = torch.exp(-(C / beta))
    # [n, n]

    for _ in range(max_iter):
        Q = A * T
        # [n, n]

        for _ in range(L):
            d = 1 / n / (Q @ sigma)
            sigma = 1 / n / (Q.T @ d)

        d1 = torch.diag(d.squeeze(1))
        d2 = torch.diag(sigma.squeeze(1))
        T = d1 * Q * d2

    loss = torch.sum(T * C)

    return loss


def process_sent(sent):
    sent = re.sub(r"(\[PAD\]|\[CLS\]|\[SEP\]|\[UNK\]|\[MASK\])", "", sent).strip()
    sent = re.sub(r"\s{2,}", " ", sent)

    return sent


def get_scores(outputs, eps=10e-8):
    n_samples = 0
    bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
    for pair in outputs:
        pred = process_sent(pair["pred"])
        ref = process_sent(pair["trg"])

        if pred == "":
            continue

        try:
            bleu_1_ = sentence_bleu([ref.split()], pred.split(), weights=(1, 0, 0, 0))
            bleu_4_ = sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25))
            meteor_ = meteor_score([ref], pred)
            rouge_l_ = Rouge().get_scores(ref, pred, avg=True)["rouge-l"]["f"]
        except ValueError:
            pass

        bleu_1 += bleu_1_ if bleu_1_ > eps else 0
        bleu_4 += bleu_4_ if bleu_4_ > eps else 0
        meteor += meteor_ if meteor_ > eps else 0
        rouge_l += rouge_l_ if rouge_l_ > eps else 0

        n_samples += 1

    return (
        bleu_1 / n_samples if n_samples > 0 else 0,
        bleu_4 / n_samples if n_samples > 0 else 0,
        meteor / n_samples if n_samples > 0 else 0,
        rouge_l / n_samples if n_samples > 0 else 0,
    )


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


class TransEncoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim=2048, dropout=0.1):
        super(TransEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, q, k, v, attn_mask=None):
        output = q
        attention = None
        for encoder in self.encoder_layers:
            output, attention = encoder(output, k, v, attn_mask=attn_mask)

        return output


class Loss(nn.Module):
    """
    Computes nll loss (Eq. (6)), coverage loss (Eq. (12)),
    and the composite loss function that combines the two (Eq. (13)).
    """

    def __init__(self, use_coverage, use_ot_loss, encoder=None, gamma1=1, gamma2=0.1, pad_id=0):
        super().__init__()
        self.use_coverage = use_coverage
        self.use_ot_loss = use_ot_loss
        self.encoder = encoder
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.pad_id = 0

    def nll_loss(self, output, target):
        """
        Negative log likelihood of target word - Eq. (6)
        Args:
            output: predicted probs from each timestep      [B x V_x T]
            target: answer ids using extended vocab         [B x T]

        Returns:
            loss: nll loss value; averaged over batch & timestep
        """
        output = torch.log(output)
        loss = F.nll_loss(output, target, ignore_index=self.pad_id, reduction="mean")
        return loss

    def cov_loss(self, attn_dist, coverage, dec_pad_mask, dec_len):
        """
        Coverage loss at timestep t - Eq. (12)
        Args:
            attn_dist: attention distribution from all timesteps            [B x L x T]
            coverage: sum of previous attn dist's from all timesteps        [B x L x T]
            dec_pad_mask: target sequence padding masks [PAD] -> True       [B x T]
            dec_len: target sequence lengths                                [B]

        Returns:
            loss: coverage loss value; averaged over batch & timestep
        """
        min_val = torch.min(attn_dist, coverage)  # [B x L x T]
        loss = torch.sum(min_val, dim=1)  # [B x T]

        # ignore loss from [PAD] tokens
        loss = loss.masked_fill_(dec_pad_mask, 0.0)
        avg_loss = torch.sum(loss) / torch.sum(dec_len)
        return avg_loss

    def ot_loss(self, pred, trg):
        return ipot(pred, trg, max_iter=400)

    def forward(self, output_mle, cov, attn_dist, trg, dec_mask, dec_len):
        nll_loss = self.nll_loss(output=output_mle, target=trg)

        if self.use_coverage:
            cov_loss = self.cov_loss(attn_dist, cov, dec_mask, dec_len)
        else:
            cov_loss = 0

        if self.use_ot_loss:
            # TODO: Modify later: remember output_mle takes tokens not in self.embedding (oov)
            trg = self.encoder(trg)
            pred = (
                torch.softmax(output_mle, dim=-1) @ self.encoder.embedding.word_embeddings.weight
            )
            ot_loss = ipot(pred, trg, max_iter=400)
        else:
            ot_loss = 0

        final_loss = nll_loss + self.gamma1 * cov_loss + self.gamma2 + ot_loss
        return final_loss
