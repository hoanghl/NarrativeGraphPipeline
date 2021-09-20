import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamodules.dataset import Vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    def __init__(self, d_embd, d_hid, path_pretrained, path_vocab, num_layers, dropout):
        super().__init__()

        emb_vecs = self.load_embeddings(d_embd, path_vocab, path_pretrained)
        self.embedding = nn.Embedding.from_pretrained(
            emb_vecs, freeze=False, padding_idx=Vocab(path_vocab).pad_id
        )
        self.biLSTM = nn.LSTM(
            input_size=d_embd,
            hidden_size=d_hid // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

    def load_embeddings(self, d_embd, path_vocab, path_pretrained):
        with open(path_pretrained.replace("[d_embd]", f"{d_embd}"), "rt") as f:
            full_content = f.read().strip().split("\n")

        glove = {}
        for entry in full_content:
            tmp = entry.split(" ")
            i_word = tmp[0]
            i_embeddings = [float(val) for val in tmp[1:]]
            glove[i_word] = np.array(i_embeddings)

        with open(path_vocab, "r") as f:
            vocab = json.load(f)

        emb_vecs = []
        for word in vocab:
            try:
                w_emb = torch.from_numpy(glove[word])
            except KeyError:
                w_emb = torch.rand((d_embd, 1))
                nn.init.kaiming_normal_(w_emb, mode="fan_out")
                w_emb = w_emb.squeeze()

            emb_vecs.append(w_emb)

        emb_vecs = torch.stack(emb_vecs)

        return emb_vecs

    def forward(self, s_ids, s_masks):
        # s_ids: [bz, l]
        # s_masks: [bz, l]

        bz, l = s_ids.size()

        s = self.embedding(s_ids).float()
        # [bz, l, d_embd]

        packed = pack_padded_sequence(
            s, s_masks.sum(-1).cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.biLSTM(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        # [bz, l_, d_hid]

        ## Pad inf
        d_hid = output.size(-1)
        pad_ = torch.zeros(
            (bz, l - output.size(1), d_hid),
            dtype=output.dtype,
            device=output.device,
        )
        output = torch.cat((output, pad_), 1)
        # [bz, l, d_hid]

        return output

    def embed_only(self, s_ids):
        # s_ids: [bz, l_]

        return self.embedding(s_ids).float()


class IntrospectiveAlignmentLayer(nn.Module):
    def __init__(self, batch_size, lc, d_hid, dropout, block, device):
        super().__init__()

        self.lin1 = nn.Sequential(nn.Linear(d_hid, d_hid), nn.Tanh(), nn.Dropout(dropout))
        self.lin2 = nn.Sequential(nn.Linear(4 * d_hid, 4 * d_hid), nn.Tanh(), nn.Dropout(dropout))
        self.biLSTM_attn = nn.LSTM(
            8 * d_hid, d_hid, num_layers=5, batch_first=True, bidirectional=True
        )

        self.mask = torch.zeros((1, lc, lc), dtype=torch.float, device=device)
        for i in range(lc):
            for j in range(lc):
                if abs(i - j) <= block:
                    self.mask[0, i, j] = 1
        self.mask = self.mask.repeat((batch_size, 1, 1))

    def forward(self, Hq, Hc):
        # Hq: [bz, lq, d_hid]
        # Hc: [bz, lc, d_hid]

        bz = Hq.size(0)

        # Introspective Alignment
        Hq, Hc = self.lin1(Hq), self.lin1(Hc)
        # [bz, l_, d_hid]

        E = Hc @ Hq.transpose(-1, -2)
        # E: [bz, lc, lq]
        A = E.softmax(-1) @ Hq
        # A: [bz, lc, d_hid]

        # Reasoning over alignments
        tmp = torch.cat((A, Hc, A - Hc, A * Hc), dim=-1)
        # [bz, lc, 4*d_hid]
        G = self.lin2(tmp)
        # [bz, lc, 4*d_hid]

        G = G @ G.transpose(-1, -2)
        G = G * self.mask.type_as(G)[:bz]
        # [bz, lc, lc]

        # Local BLock-based Self-Attention
        B = G.softmax(-1) @ tmp
        # B: [bz, lc, 4*d_hid]

        Y = torch.cat((B, tmp), dim=-1)
        # [bz, lc, 8*d_hid]
        Y = self.biLSTM_attn(Y)[0]
        # [bz, lc, 2*d_hid]

        return Y


class AttnPooling(nn.Module):
    def __init__(self, d_hid, dropout):
        super().__init__()

        self.lin_att1 = nn.Linear(d_hid, d_hid, bias=False)
        self.lin_att2 = nn.Linear(d_hid, d_hid, bias=False)
        self.ff_attn = nn.Sequential(
            nn.Linear(d_hid, d_hid), nn.Tanh(), nn.Dropout(dropout), nn.Linear(d_hid, 1)
        )

    def forward(self, ai, q, q_masks):
        # ai: [bz, d_hid]
        # q: [bz, lq, d_hid]
        # q_masks: [bz, lq]

        a = self.ff_attn(self.lin_att1(q) + self.lin_att2(ai).unsqueeze(1)).squeeze(-1)
        # [bz, lq]
        a = a.float().masked_fill_(q_masks == 0, float("-inf")).type_as(q)
        a = a.softmax(-1)

        h = (q * a.unsqueeze(-1)).sum(dim=1)
        # [bz, d_hid]

        return h


class NonLinear(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()

        self.ff = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Dropout(dropout))

    def forward(self, X):
        return self.ff(X)


class PGN(nn.Module):
    def __init__(self, la, d_embd, d_hid, d_vocab, num_layers, dropout, path_vocab, embedding):
        super().__init__()

        self.la = la
        vocab = Vocab(path_vocab)
        self.sos_id = vocab.sos_id
        self.eos_id = vocab.eos_id
        self.d_vocab = d_vocab

        self.embedding = embedding
        self.attn_pooling = AttnPooling(d_hid, dropout)
        self.ff1 = NonLinear(d_hid, dropout)
        self.ff2 = self.ff = nn.Sequential(
            NonLinear(2 * d_hid, dropout), nn.Linear(2 * d_hid, d_hid)
        )
        self.ff3 = nn.Sequential(NonLinear(d_hid, dropout), nn.Linear(d_hid, 1))
        self.lin1 = nn.Linear(d_embd, d_hid, bias=False)
        self.lstm = nn.LSTM(2 * d_hid, d_hid, num_layers, batch_first=True, dropout=dropout)
        self.lin_pgn = nn.Linear(d_hid, d_vocab)
        self.lin_pgn1 = nn.Linear(d_hid, 1)
        self.lin_pgn2 = nn.Linear(d_hid, 1)
        self.lin_pgn3 = nn.Linear(d_hid, 1)

    def do_train(self, Y, q, q_masks, a, c_ids_ext, c_masks):
        # Y: [bz, lc, 2 * d_hid]
        # q: [bz, lq, d_hid]
        # q_masks: [bz, lq]
        # a: [bz, la, d_embd]
        # c_ids_ext: [bz, lc]

        Y = self.ff2(Y)
        # [bz, lc, d_hid]
        a = self.lin1(a)
        # [bz, la, d_hid]

        outputs = []
        h, c = None, None
        for i in range(self.la):
            ai = a[:, i]

            ## Attention over Y
            h_ = self.ff1(h.transpose(0, 1).sum(dim=1)).unsqueeze(1) if torch.is_tensor(h) else 0
            at = self.ff3(h_ + Y + self.attn_pooling(ai, q, q_masks).unsqueeze(1)).squeeze(-1)
            # [bz, lc]
            ## Mask padding positions
            at = at.float().masked_fill_(c_masks == 0, float("-inf")).type_as(q)
            at = at.softmax(-1)
            y = (at.unsqueeze(-1) * Y).sum(dim=1)
            # [bz, d_hid]

            ## Pass over LSTM
            _, (h, c) = (
                self.lstm(torch.cat((y, ai), -1).unsqueeze(1), (h, c))
                if torch.is_tensor(h)
                else self.lstm(torch.cat((y, ai), -1).unsqueeze(1))
            )
            # h, c: [num_layers, bz, d_hid]

            ## PGN
            h_, c_ = h.transpose(0, 1).sum(dim=1), c.transpose(0, 1).sum(dim=1)
            pt = F.sigmoid(self.lin_pgn1(c_) + self.lin_pgn2(h_) + self.lin_pgn3(y))
            # [bz, 1]

            vt = self.lin_pgn(h_)
            # [bz, d_vocab]
            vt = vt.softmax(-1)
            vt = (1 - pt) * vt

            at = pt * at

            ## Padd zeros to vt for OOV positions
            extra_zeros = torch.zeros_like(at)
            extended_vt = torch.cat((vt, extra_zeros), dim=-1)
            # [bz, d_vocab + lc]

            wt = extended_vt.scatter_add(-1, index=c_ids_ext, src=at)
            # [bz, d_vocab + lc]

            outputs.append(wt.unsqueeze(1))

        output_mle = torch.cat(outputs, 1)
        # [b, la, d_vocab + lc]

        return output_mle

    def do_predict(self, Y, q, q_masks, c_ids_ext, c_masks):
        # Y: [bz, lc, 2 * d_hid]
        # q: [bz, lq, d_hid]
        # c_ids: [bz, lc]

        bz = Y.size(0)

        Y = self.ff2(Y)
        # [bz, lc, d_hid]
        outputs = torch.zeros(bz, self.la, dtype=torch.long, device=Y.device)

        for b in range(bz):
            h, c = 0, 0
            q_, q_masks_, c_ids_ext_, c_masks_, Y_ = (
                q[b : b + 1],
                q_masks[b : b + 1],
                c_ids_ext[b : b + 1],
                c_masks[b : b + 1],
                Y[b : b + 1],
            )
            for i in range(self.la):
                ## Embd token generated in last step
                last_tok = (
                    outputs[b : b + 1, i - 1 : i]
                    if i > 0
                    else torch.full(
                        (1, 1), self.sos_id, dtype=c_ids_ext.dtype, device=c_ids_ext.device
                    )
                )
                # [1, 1]
                last_tok = self.embedding.embed_only(last_tok).squeeze(1)
                # [1, d_embd]
                last_tok = self.lin1(last_tok)
                # [1, d_hid]

                ## Attention over Y
                h_ = (
                    self.ff1(h.transpose(0, 1).sum(dim=1)).unsqueeze(1)
                    if torch.is_tensor(h)
                    else 0
                )
                at = self.ff3(
                    h_ + Y_ + self.attn_pooling(last_tok, q_, q_masks_).unsqueeze(1)
                ).squeeze(-1)
                # [1, lc]
                ## Mask padding positions
                at = at.float().masked_fill_(c_masks_ == 0, float("-inf")).type_as(q)
                at = at.softmax(-1)
                y = (at.unsqueeze(-1) * Y_).sum(dim=1)
                # [1, d_hid]

                ## Pass over LSTM
                _, (h, c) = (
                    self.lstm(torch.cat((y, last_tok), -1).unsqueeze(1), (h, c))
                    if torch.is_tensor(h)
                    else self.lstm(torch.cat((y, last_tok), -1).unsqueeze(1))
                )
                # h, c: [num_layers, 1, d_hid]

                ## PGN
                h_, c_ = h.transpose(0, 1).sum(dim=1), c.transpose(0, 1).sum(dim=1)
                pt = F.sigmoid(self.lin_pgn1(c_) + self.lin_pgn2(h_) + self.lin_pgn3(y))
                # [1, 1]

                vt = self.lin_pgn(h_)
                # [1, d_vocab]
                vt = vt.softmax(-1)
                vt = (1 - pt) * vt

                at = pt * at

                ## Padd zeros to vt for OOV positions
                extra_zeros = torch.zeros_like(at)
                extended_vt = torch.cat((vt, extra_zeros), dim=-1)
                # [1, d_vocab + lc]

                wt = extended_vt.scatter_add(-1, index=c_ids_ext_, src=at)
                # [1, d_vocab + lc]

                pred_tok = wt.argmax(-1)
                # [1]
                outputs[b, i] = pred_tok
                if pred_tok == self.eos_id:
                    break

        return outputs
