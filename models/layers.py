import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from transformers import BertModel, logging
from transformers.models.bert.tokenization_bert import BertTokenizer

logging.set_verbosity_error()


class Embedding(torch_nn.Module):
    def __init__(self, d_hid, d_bert, path_pretrained):
        super().__init__()

        tokenizer = BertTokenizer.from_pretrained(path_pretrained)
        self.pad_id = tokenizer.pad_token_id

        self.embedding = BertModel.from_pretrained(path_pretrained)
        self.lin = torch_nn.Linear(d_bert, d_hid)

    def _get_padding_mask(self, ids):
        s_l = ids.size(0)
        l = (ids != self.pad_id).sum()
        ones = torch.ones((s_l, s_l), device=ids.device)
        mask = ones.tril()
        mask[:, :l] = 1
        padding_mask = (ids != self.pad_id).float()
        mask *= padding_mask
        mask *= padding_mask.unsqueeze(1)
        return mask

    def embed_qc(self, q_ids, c_ids):
        # q_ids: [bz, lq]
        # c_ids: [bz, nc, lc]
        # a1_ids: [bz, la]
        # a2_ids: [bz, la]

        bz = c_ids.size(0)

        ## Create mask matrix
        q_masks = []
        for q in q_ids:
            q_msk = self._get_padding_mask(q)
            q_masks.append(q_msk.unsqueeze(0))

        q_masks = torch.cat(q_masks, 0)

        c_ids = c_ids.view(-1, c_ids.size(-1))
        c_masks_, c_l = [], []
        for c in c_ids:
            c_l.append((c != self.pad_id).sum().unsqueeze(0))
            c_masks_.append(self._get_padding_mask(c).unsqueeze(0))
        c_masks_ = torch.cat(c_masks_, 0)
        c_l = torch.cat(c_l, 0)

        ## Embed
        q = self.embedding(q_ids, q_masks)[0]
        c_ = self.embedding(c_ids, c_masks_)[0]

        ## Merge para into single one
        c, c_masks, c_ids_, c_ids_mask_ = [], [], [], []
        for ci, ci_l, cid in zip(c_, c_l, c_ids):
            c.append(ci[:ci_l])
            c_masks.append(ci[ci_l:])
            c_ids_.append(cid[:ci_l])
            c_ids_mask_.append(cid[ci_l:])
        c = torch.cat(c + c_masks, 0).view(bz, -1, q.size(-1))
        # [bz, nc*lc, d_bert]
        c_ids = torch.cat(c_ids_ + c_ids_mask_, 0).view(bz, -1)
        # [bz, nc*lc]

        ## Move to new dimension
        q, c = self.lin(q), self.lin(c)
        # [bz, l_, d_hid]

        return q, c, c_ids

    def embed_a(self, a_ids):
        # a_ids: [b, la]

        ## Create mask matrix
        a_masks = []
        for a in a_ids:
            a_msk = self._get_padding_mask(a)
            a_masks.append(a_msk.unsqueeze(0))
        a_masks = torch.cat(a_masks, 0)

        ## Embed
        a = self.embedding(a_ids, a_masks)[0]

        ## Move to new dimension
        a = self.lin(a)
        # [bz, la, d_hid]

        return a

    def embed_token(self, ids):
        # a_ids: [b_, 1]

        # Embed
        embd = self.embedding.embeddings.word_embeddings(ids)

        ## Move to new dimension
        embd = self.lin(embd)
        # [b_, 1, d_hid]

        return embd


class IntrospectiveAlignmentLayer(torch_nn.Module):
    def __init__(self, batch_size, nc, lc, d_hid, dropout, block, device):
        super().__init__()

        self.lin1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid), torch_nn.Tanh(), torch_nn.Dropout(dropout)
        )
        self.lin2 = torch_nn.Sequential(
            torch_nn.Linear(4 * d_hid, 4 * d_hid), torch_nn.Tanh(), torch_nn.Dropout(dropout)
        )
        self.biLSTM_attn = torch_nn.LSTM(
            8 * d_hid, d_hid, num_layers=5, batch_first=True, bidirectional=True
        )

        l = nc * lc
        self.mask = torch.zeros((1, l, l), dtype=torch.float, device=device)
        for i in range(l):
            for j in range(l):
                if abs(i - j) <= block:
                    self.mask[0, i, j] = 1
        self.mask = self.mask.repeat((batch_size, 1, 1))

    def forward(self, Hq, Hc):
        # Hq: [bz, lq, d_hid]
        # Hc: [bz, nc*lc, d_hid]

        # l = nc * lc
        bz = Hq.size(0)

        # Introspective Alignment
        Hq, Hc = self.lin1(Hq), self.lin1(Hc)
        # [bz, l_, d_hid]

        E = Hc @ Hq.transpose(-1, -2)
        # E: [bz, l, lq]
        A = E.softmax(-1) @ Hq
        # A: [bz, l, d_hid]

        # Reasoning over alignments
        tmp = torch.cat((A, Hc, A - Hc, A * Hc), dim=-1)
        # [bz, l, 4*d_hid]
        G = self.lin2(tmp)
        # [bz, l, 4*d_hid]

        G = G @ G.transpose(-1, -2)
        G = G * self.mask.type_as(G)[:bz]
        # [bz, l, l]

        # Local BLock-based Self-Attention
        B = G.softmax(-1) @ tmp
        # B: [bz, l, 4*d_hid]

        Y = torch.cat((B, tmp), dim=-1)
        # [bz, l, 8*d_hid]
        Y = self.biLSTM_attn(Y)[0]
        # [bz, l, 2*d_hid]

        return Y


class AttnPooling(torch_nn.Module):
    def __init__(self, d_hid):
        super().__init__()

        self.lin_att1 = torch_nn.Linear(d_hid, d_hid, bias=False)
        self.lin_att2 = torch_nn.Linear(d_hid, d_hid, bias=False)
        self.ff_attn = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Tanh(),
            torch_nn.Dropout(),
            torch_nn.Linear(d_hid, 1),
            torch_nn.Softmax(-1),
        )

    def forward(self, ai, q):
        # q: [bz, d_hid]
        # ai: [bz, lq, d_hid]

        a = self.ff_attn(self.lin_att1(q) + self.lin_att2(ai).unsqueeze(1))
        # [bz, lq, 1]

        h = (q * a).sum(dim=1)
        # [bz, d_hid]

        return h


class NonLinear(torch_nn.Module):
    def __init__(self, d, dropout):
        super().__init__()

        self.ff = torch_nn.Sequential(
            torch_nn.Linear(d, d), torch_nn.Tanh(), torch_nn.Dropout(dropout)
        )

    def forward(self, X):
        return self.ff(X)


class PGN(torch_nn.Module):
    def __init__(self, la, d_hid, d_vocab, num_layers_lstm, dropout, path_pretrained, embedding):
        super().__init__()

        tokenizer = BertTokenizer.from_pretrained(path_pretrained)
        self.la = la
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.d_vocab = d_vocab

        self.embedding = embedding
        self.attn_pooling = AttnPooling(d_hid)
        self.ff1 = NonLinear(d_hid, dropout)
        self.ff2 = self.ff = torch_nn.Sequential(
            NonLinear(2 * d_hid, dropout), torch_nn.Linear(2 * d_hid, d_hid)
        )
        self.ff3 = torch_nn.Sequential(
            NonLinear(d_hid, dropout), torch_nn.Linear(d_hid, 1), torch_nn.Softmax(-1)
        )
        self.lstm = torch_nn.LSTM(
            2 * d_hid, d_hid, num_layers_lstm, batch_first=True, dropout=dropout
        )
        self.lin_pgn = torch_nn.Linear(d_hid, d_vocab)
        self.lin_pgn1 = torch_nn.Linear(d_hid, 1)
        self.lin_pgn2 = torch_nn.Linear(d_hid, 1)
        self.lin_pgn3 = torch_nn.Linear(d_hid, 1)

    def do_train(self, Y, q, a, c_ids):
        # Y: [bz, nc * lc, 2 * d_hid]
        # q: [bz, lq, d_hid]
        # a: [bz, la, d_hid]
        # c_ids: [bz, nc*lc]

        # l = nc * lc

        Y = self.ff2(Y)
        # [bz, l, d_hid]
        outputs = []
        h, c = None, None
        for i in range(self.la):
            ai = a[:, i]

            ## Attention over Y
            h_ = self.ff1(h.transpose(0, 1).sum(dim=1)).unsqueeze(1) if torch.is_tensor(h) else 0
            at = self.ff3(h_ + Y + self.attn_pooling(ai, q).unsqueeze(1))
            # [bz, l, 1]
            y = (at * Y).sum(dim=1)
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
            vt = self.lin_pgn(h_)
            # [bz, d_vocab]
            pt = F.sigmoid(self.lin_pgn1(c_) + self.lin_pgn2(h_) + self.lin_pgn3(y))
            # [bz, 1]
            wt = (1 - pt) * vt
            at = torch.scatter_add(torch.zeros_like(vt), -1, c_ids, at.squeeze(-1)).softmax(-1)
            wt = wt + pt * at
            # [bz, d_vocab]

            outputs.append(wt.unsqueeze(-1))

        output_mle = torch.cat(outputs, -1)
        # [b, d_vocab, la]

        return output_mle

    def do_predict(self, Y, q, c_ids):
        # Y: [bz, nc * lc, 2 * d_hid]
        # q: [bz, lq, d_hid]
        # a: [bz, l_, d_hid]
        # c_ids: [bz, nc*lc]

        bz = Y.size(0)

        Y = self.ff2(Y)
        # [bz, l, d_hid]
        outputs = torch.zeros(bz, self.la, dtype=torch.long, device=Y.device)

        for b in range(bz):
            h, c = 0, 0
            q_, c_ids_, Y_ = q[b : b + 1], c_ids[b : b + 1], Y[b : b + 1]
            for i in range(self.la):
                ## Embd token generated in last step
                last_tok = (
                    outputs[b : b + 1, i - 1 : i]
                    if i > 0
                    else torch.full((1, 1), self.cls_id, dtype=c_ids.dtype, device=c_ids.device)
                )
                # [1, 1]
                last_tok = self.embedding.embed_token(last_tok).squeeze(1)
                # [1, d_hid]

                ## Attention over Y
                h_ = (
                    self.ff1(h.transpose(0, 1).sum(dim=1)).unsqueeze(1)
                    if torch.is_tensor(h)
                    else 0
                )
                at = self.ff3(h_ + Y_ + self.attn_pooling(last_tok, q_).unsqueeze(1))
                # [1, l, 1]
                y = (at * Y_).sum(dim=1)
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
                vt = self.lin_pgn(h_)
                # [1, d_vocab]
                pt = F.sigmoid(self.lin_pgn1(c_) + self.lin_pgn2(h_) + self.lin_pgn3(y))
                # [1, 1]

                wt = (1 - pt) * vt
                at = torch.scatter_add(torch.zeros_like(vt), -1, c_ids_, at.squeeze(-1)).softmax(
                    -1
                )
                wt = wt + pt * at
                # [1, d_vocab]

                pred_tok = wt.argmax(-1)
                # [1]
                outputs[b, i] = pred_tok
                if pred_tok == self.sep_id:
                    break

        return outputs
