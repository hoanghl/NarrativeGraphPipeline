import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, d_embd, d_hid, path_pretrained, num_layers, dropout, vocab):
        super().__init__()

        self.vocab = vocab

        emb_vecs = self.load_embeddings(d_embd, path_pretrained)
        self.embedding = nn.Embedding.from_pretrained(
            emb_vecs, freeze=False, padding_idx=vocab.pad()
        )
        self.biLSTM = nn.LSTM(
            input_size=d_embd,
            hidden_size=d_hid,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def load_embeddings(self, d_embd, path_pretrained):
        with open(path_pretrained.replace("[d_embd]", f"{d_embd}"), "rt") as f:
            full_content = f.read().strip().split("\n")

        emb_vecs = []
        glove = {}
        for entry in full_content:
            tmp = entry.split(" ")
            i_word = tmp[0]
            i_embeddings = [float(val) for val in tmp[1:]]
            glove[i_word] = np.array(i_embeddings)

        for w in self.vocab._id_to_word:
            try:
                w_emb = torch.from_numpy(glove[w])

            except KeyError:
                w_emb = torch.rand([d_embd, 1])
                nn.init.kaiming_normal_(w_emb, mode="fan_out")

            emb_vecs.append(w_emb)
        emb_vecs = list(map(lambda x: x.squeeze(), emb_vecs))
        emb_vecs = torch.stack(emb_vecs)

        return emb_vecs

    def forward(self, s_ids, s_len):
        # s_ids: [bz, l]
        # s_masks: [bz]

        s = self.embedding(s_ids).float()
        # [bz, l, d_embd]

        packed = pack_padded_sequence(s, s_len.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.biLSTM(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        # [bz, l_, d_hid * 2]

        return output

    def embed_only(self, s_ids):
        # s_ids: [bz, l_]

        return self.embedding(s_ids).float()


class IntrospectiveAlignmentLayer(nn.Module):
    def __init__(self, d, dropout, block):
        super().__init__()

        self.block = block

        self.lin1 = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Dropout(dropout))
        self.lin2 = nn.Sequential(nn.Linear(4 * d, 4 * d), nn.Tanh(), nn.Dropout(dropout))
        self.biLSTM_attn = nn.LSTM(8 * d, d, num_layers=5, batch_first=True, bidirectional=True)

        self.mask = None
        self.lin3 = nn.Linear(d * 2, d, bias=False)

    def forward(self, Hq, Hc):
        # Hq: [bz, lq, d]
        # Hc: [bz, lc, d]

        bz, lc, _ = Hc.size()

        if not torch.is_tensor(self.mask) or self.mask.size(0) != Hq.size(0):
            self.mask = torch.zeros((1, lc, lc), dtype=Hc.dtype, device=Hc.device)
            for i in range(lc):
                for j in range(lc):
                    if abs(i - j) <= self.block:
                        self.mask[0, i, j] = 1
            self.mask = self.mask.repeat((bz, 1, 1))

        bz = Hq.size(0)

        # Introspective Alignment
        Hq, Hc = self.lin1(Hq), self.lin1(Hc)
        # [bz, l_, d]

        E = Hc @ Hq.transpose(-1, -2)
        # E: [bz, lc, lq]
        A = E.softmax(-1) @ Hq
        # A: [bz, lc, d]

        # Reasoning over alignments
        tmp = torch.cat((A, Hc, A - Hc, A * Hc), dim=-1)
        # [bz, lc, 4*d]
        G = self.lin2(tmp)
        # [bz, lc, 4*d]

        G = G @ G.transpose(-1, -2)
        G = G * self.mask.type_as(G)[:bz]
        # [bz, lc, lc]

        # Local BLock-based Self-Attention
        B = G.softmax(-1) @ tmp
        # B: [bz, lc, 4*d]

        Y = torch.cat((B, tmp), dim=-1)
        # [bz, lc, 8*d]
        Y = self.biLSTM_attn(Y)[0]
        # [bz, lc, 2*d]

        Y = self.lin3(Y)

        return Y


class AttnPooling(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.lin_att1 = nn.Linear(d, d, bias=False)
        self.lin_att2 = nn.Linear(d, d)
        self.ff_attn = nn.Sequential(nn.Tanh(), nn.Linear(d, 1))

    def forward(self, ai, q, q_masks):
        # ai: [bz, d_hid * 2]
        # q: [bz, lq, d_hid * 2]
        # q_masks: [bz, lq]

        a = self.ff_attn(self.lin_att1(q) + self.lin_att2(ai).unsqueeze(1)).squeeze(-1)
        # [bz, lq]
        a = a.float().masked_fill_(q_masks, float("-inf")).type_as(q)
        a = a.softmax(-1)

        h = (a.unsqueeze(1) @ q).sum(1)
        # [bz, d_hid]

        return h


class NonLinear(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()

        self.ff = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Dropout(dropout))

    def forward(self, X):
        return self.ff(X)


class PGN(nn.Module):
    def __init__(self, la, d_embd, d, d_vocab, dropout, embedding, vocab):
        super().__init__()

        self.la = la
        self.vocab = vocab
        self.sos_id = vocab.start()
        self.eos_id = vocab.stop()
        self.d_vocab = d_vocab

        self.encoder = embedding
        self.w_c = nn.Linear(1, d, bias=False)
        self.attn_pooling = AttnPooling(d)
        self.ff = nn.Sequential(NonLinear(d, dropout), nn.Linear(d, 1))
        self.lin1 = nn.Linear(d_embd, d, bias=False)
        self.lstm = nn.LSTMCell(d * 2, d)
        self.lin_pgn = nn.Linear(d, d_vocab)
        self.lin_pgn1 = nn.Linear(d, 1, bias=False)
        self.lin_pgn2 = nn.Linear(d, 1, bias=False)
        self.lin_pgn3 = nn.Linear(d, 1, bias=False)

    def forward(self, ai, h, c, cov, Y, q, q_masks, c_ids_ext, c_masks, max_oov_len):
        bz = Y.size(0)

        ## Attention over Y
        at = self.ff(
            h.unsqueeze(1)
            + Y
            + self.attn_pooling(ai, q, q_masks).unsqueeze(1)
            + self.w_c(cov.unsqueeze(-1))
        ).squeeze(-1)
        # [bz, lc]
        ## Mask padding positions
        at = at.float().masked_fill_(c_masks, float("-inf")).type_as(q)
        at = at.softmax(-1)
        y = (at.unsqueeze(1) @ Y).sum(dim=1)
        # [bz, d]

        cov = cov + at

        ## Pass over LSTM
        h, c = self.lstm(torch.cat((y, ai), -1), (h, c))
        # h, c: [bz, d]

        ## PGN
        pt = F.sigmoid(self.lin_pgn1(c) + self.lin_pgn2(h) + self.lin_pgn3(y))
        # [bz, 1]

        vt = self.lin_pgn(h)
        # [bz, d_vocab]
        vt = vt.softmax(-1)
        vt = (1 - pt) * vt

        at = pt * at

        ## Padd zeros to vt for OOV positions
        extra_zeros = torch.zeros((bz, max_oov_len), device=vt.device, dtype=vt.dtype)
        extended_vt = torch.cat((vt, extra_zeros), dim=-1)
        # [bz, d_vocab + max_oov_len]

        wt = extended_vt.scatter_add(-1, index=c_ids_ext, src=at)
        # [bz, d_vocab + max_oov_len]

        return wt, cov, at

    def do_train(self, Y, q, q_masks, a, c_ids_ext, c_masks, max_oov_len):
        # Y: [bz, lc, 2 * d_hid]
        # q: [bz, lq, 2 * d_hid]
        # q_masks: [bz, lq]
        # a: [bz, la, d_embd]
        # c_ids_ext: [bz, lc]

        # d = d_hid * 2

        bz, _, d = Y.size()

        a = self.lin1(a)
        # [bz, la, d_hid]

        outputs, coverage, attn_dists = [], [], []
        cov = torch.zeros_like(c_ids_ext).float()  # [B x L]
        h, c = torch.zeros(bz, d, device=Y.device, dtype=Y.dtype), torch.zeros(
            bz, d, device=Y.device, dtype=Y.dtype
        )
        for i in range(self.la):
            ai = a[:, i]

            wt, cov, at = self(ai, h, c, cov, Y, q, q_masks, c_ids_ext, c_masks, max_oov_len)

            outputs.append(wt.unsqueeze(-1))
            coverage.append(cov.unsqueeze(-1))
            attn_dists.append(at.unsqueeze(-1))

        output_mle = torch.cat(outputs, -1)
        # [bz, d_vocab + max_oov_len, la]
        coverage = torch.cat(coverage, -1)
        # [bz, lc, la]
        attn_dists = torch.cat(attn_dists, -1)
        # [bz, lc, la]

        return output_mle, coverage, attn_dists

    def do_predict(self, Y, q, q_masks, c_ids_ext, c_masks, max_oov_len):
        # Y: [bz, lc, 2 * d_hid]
        # q: [bz, lq, d_hid]
        # c_ids: [bz, lc]

        _, lc, d = Y.size()

        # [bz, lc, d_hid]
        output_mle = torch.zeros(
            1, self.d_vocab + max_oov_len, self.la, dtype=Y.dtype, device=Y.device
        )
        coverage = torch.zeros(1, lc, self.la, dtype=torch.long, device=Y.device)
        attn_dists = torch.zeros_like(coverage)

        cov = torch.zeros_like(c_ids_ext).float()  # [B x L]
        h, c = torch.zeros(1, d, device=Y.device, dtype=Y.dtype), torch.zeros(
            1, d, device=Y.device, dtype=Y.dtype
        )
        preds = []
        last_tok = torch.full((1, 1), self.sos_id, dtype=c_ids_ext.dtype, device=c_ids_ext.device)
        for i in range(self.la):
            ## Embd token generated in last step
            ai = self.encoder.embed_only(last_tok).squeeze(1)
            # [1, d_embd]
            ai = self.lin1(ai)
            # [1, d_hid]

            wt, cov, at = self(ai, h, c, cov, Y, q, q_masks, c_ids_ext, c_masks, max_oov_len)
            # wt: [1, d_vocab + max_oov_len]
            # cov: [1, lc]
            # at: [1, lc]

            output_mle[:, :, i] = wt
            coverage[:, :, i] = cov
            attn_dists[:, :, i] = at

            last_tok = wt.argmax(-1).unsqueeze(-1)
            if last_tok == self.eos_id:
                break
            if last_tok >= self.d_vocab:
                last_tok = torch.tensor(
                    [[self.vocab.unk()]], dtype=last_tok.dtype, device=last_tok.device
                )

            preds.append(last_tok)

        preds = torch.cat(preds, -1)

        return preds, output_mle, coverage, attn_dists
