import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    def __init__(self, d_embd, d_hid, path_pretrained, num_layers, dropout, vocab):
        super().__init__()

        self.vocab = vocab

        emb_vecs = self.load_embeddings(d_embd, path_pretrained)
        self.embedding = nn.Embedding.from_pretrained(
            emb_vecs, freeze=False, padding_idx=vocab.pad()
        )
        self.biLSTM = nn.LSTM(
            input_size=d_embd,
            hidden_size=d_hid // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
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

        bz, l = s_ids.size()

        s = self.embedding(s_ids).float()
        # [bz, l, d_embd]

        packed = pack_padded_sequence(s, s_len.cpu(), batch_first=True, enforce_sorted=False)
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
    def __init__(self, d_hid, dropout, block):
        super().__init__()

        self.block = block

        self.lin1 = nn.Sequential(nn.Linear(d_hid, d_hid), nn.Tanh(), nn.Dropout(dropout))
        self.lin2 = nn.Sequential(nn.Linear(4 * d_hid, 4 * d_hid), nn.Tanh(), nn.Dropout(dropout))
        self.biLSTM_attn = nn.LSTM(
            8 * d_hid, d_hid, num_layers=5, batch_first=True, bidirectional=True
        )

        self.mask = None

    def forward(self, Hq, Hc):
        # Hq: [bz, lq, d_hid]
        # Hc: [bz, lc, d_hid]

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
        self.lin_att2 = nn.Linear(d_hid, d_hid)
        self.ff_attn = nn.Sequential(nn.Tanh(), nn.Linear(d_hid, 1))

    def forward(self, ai, q, q_masks):
        # ai: [bz, d_hid]
        # q: [bz, lq, d_hid]
        # q_masks: [bz, lq]

        a = self.ff_attn(self.lin_att1(q) + self.lin_att2(ai).unsqueeze(1)).squeeze(-1)
        # [bz, lq]
        a = a.float().masked_fill_(q_masks == 0, float("-inf")).type_as(q)
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


class Attention(nn.Module):
    """
    Attention mechanism based on Bahdanau et al. (2015) - Eq. (1)(2)
    augmented with Coverage mechanism - Eq. (11)
    B : batch size
    L : source text length
    H : encoder hidden state dimension
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Eq. (1)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)  # v
        self.enc_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)  # W_h
        self.dec_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)  # W_s, b_attn

    def forward(self, dec_input, enc_hidden, enc_pad_mask):
        """
        Args:
            dec_input: decoder hidden state             [B x H]
            coverage: coverage vector                   [B x L]
            enc_hidden: encoder hidden states           [B x L x 2H]
            enc_pad_mask: encoder padding masks         [B x L]

        Returns:
            attn_dist: attention dist'n over src tokens [B x L]
        """

        # Eq. (1)
        enc_feature = self.enc_proj(enc_hidden)  # [B x L x 2H]
        dec_feature = self.dec_proj(dec_input)  # [B x 2H]
        dec_feature = dec_feature.unsqueeze(1)  # [B x 1 x 2H]
        scores = enc_feature + dec_feature  # [B x L x 2H]

        scores = torch.tanh(scores)  # [B x L x 2H]
        scores = self.v(scores)  # [B x L x 1]
        scores = scores.squeeze(-1)  # [B x L]

        # Don't attend over padding; fill '-inf' where enc_pad_mask == True
        if enc_pad_mask is not None:
            scores = (
                scores.float().masked_fill_(enc_pad_mask, float("-inf")).type_as(scores)
            )  # FP16 support: cast to float and back

        # Eq. (2)
        attn_dist = F.softmax(scores, dim=-1)  # [B x L]

        return attn_dist


class AttnDecoder(nn.Module):
    """
    Single-layer unidirectional LSTM with attention for a single timestep - Eq. (3)(4)
    B : batch size
    E : embedding size
    H : decoder hidden state dimension
    V : vocab size
    """

    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
        self.attention = Attention(hidden_dim)
        # Eq. (4)
        self.v = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)  # V, b
        self.v_out = nn.Linear(hidden_dim, vocab_size, bias=True)  # V', b'

    def forward(self, dec_input, prev_h, prev_c, Y, c_pad_mask):
        """
        Args:
            dec_input: decoder input embedding at timestep t    [B x E]
            prev_h: decoder hidden state from prev timestep     [B x H]
            prev_c: decoder cell state from prev timestep       [B x H]
            Y: encoder hidden states                   [B x L x 2H]
            c_pad_mask: encoder masks for attn computation    [B x L]
            coverage: coverage vector at timestep t - Eq. (10)  [B x L]

        Returns:
            vocab_dist: predicted vocab dist'n at timestep t    [B x V]
            attn_dist: attention dist'n at timestep t           [B x L]
            context_vec: context vector at timestep t           [B x 2H]
            hidden: hidden state at timestep t                  [B x H]
            cell: cell state at timestep t                      [B x H]
        """

        # Get this step's decoder hidden state
        hidden, cell = self.lstm(dec_input, (prev_h, prev_c))  # [B x H], [B x H]

        # Compute attention distribution over enc states
        attn_dist = self.attention(
            dec_input=hidden, enc_hidden=Y, enc_pad_mask=c_pad_mask
        )  # [B x L]

        # Eq. (3) - Sum weighted enc hidden states to make context vector
        # The context vector is used later to compute generation probability
        context_vec = torch.bmm(attn_dist.unsqueeze(1), Y)  # [B x 1 x 2H]
        context_vec = torch.sum(context_vec, dim=1)  # [B x 2H]

        # Eq. (4)
        output = self.v(torch.cat([hidden, context_vec], dim=-1))  # [B x 3H] -> [B x H]
        output = self.v_out(output)  # [B x V]
        vocab_dist = F.softmax(output, dim=-1)  # [B x V]
        return vocab_dist, attn_dist, context_vec, hidden, cell


class PGN(nn.Module):
    def __init__(self, la, d_embd, d_hid, d_vocab, num_layers, dropout, embedding, vocab):
        super().__init__()

        self.la = la
        self.vocab = vocab
        self.sos_id = vocab.start()
        self.eos_id = vocab.stop()
        self.d_vocab = d_vocab

        self.embedding = embedding
        self.attn_pooling = AttnPooling(d_hid, dropout)
        self.ff1 = NonLinear(d_hid, dropout)
        self.ff2 = self.ff = nn.Sequential(
            NonLinear(2 * d_hid, dropout), nn.Linear(2 * d_hid, d_hid)
        )
        self.ff3 = nn.Sequential(NonLinear(d_hid, dropout), nn.Linear(d_hid, 1))
        self.lin1 = nn.Linear(d_embd, d_hid, bias=False)
        self.lstm = nn.LSTMCell(2 * d_hid, d_hid)
        self.lin_pgn = nn.Linear(d_hid, d_vocab)
        self.lin_pgn1 = nn.Linear(d_hid, 1, bias=False)
        self.lin_pgn2 = nn.Linear(d_hid, 1, bias=False)
        self.lin_pgn3 = nn.Linear(d_hid, 1, bias=False)

        # Parameters specific to PGN - Eq. (8)
        self.w_h = nn.Linear(d_hid * 2, 1, bias=False)
        self.w_s = nn.Linear(d_hid, 1, bias=False)
        self.w_x = nn.Linear(d_embd, 1, bias=True)

        self.decoder = AttnDecoder(input_dim=d_embd, hidden_dim=d_hid, vocab_size=d_vocab)

    def do_train(self, Y, q, q_masks, a, c_ids_ext, c_masks, max_oov_len):
        # Y: [bz, lc, 2 * d_hid]
        # q: [bz, lq, d_hid]
        # q_masks: [bz, lq]
        # a: [bz, la, d_embd]
        # c_ids_ext: [bz, lc]

        # [bz, lc, d_hid]
        # a = self.lin1(a)
        # # [bz, la, d_hid]

        bz, _, d_hid = Y.size()
        d_hid = d_hid // 2

        outputs = []
        h, c = torch.zeros(bz, d_hid, device=Y.device, dtype=Y.dtype), torch.zeros(
            bz, d_hid, device=Y.device, dtype=Y.dtype
        )
        for i in range(self.la):
            ai = a[:, i]

            # ## Attention over Y

            # at = self.ff3(
            #     h.unsqueeze(1) + Y + self.attn_pooling(ai, q, q_masks).unsqueeze(1)
            # ).squeeze(-1)
            # # [bz, lc]
            # ## Mask padding positions
            # at = at.float().masked_fill_(c_masks == 0, float("-inf")).type_as(q)
            # at = at.softmax(-1)
            # y = (at.unsqueeze(1) @ Y).sum(dim=1)
            # # [bz, d_hid]

            # ## Pass over LSTM
            # h, c = self.lstm(torch.cat((y, ai), -1), (h, c))
            # # h, c: [bz, d_hid]

            # ## PGN
            # pt = F.sigmoid(self.lin_pgn1(c) + self.lin_pgn2(h) + self.lin_pgn3(y))
            # # [bz, 1]

            # vt = self.lin_pgn(h)
            # # [bz, d_vocab]
            # vt = vt.softmax(-1)
            # vt = (1 - pt) * vt

            # at = pt * at

            # ## Padd zeros to vt for OOV positions
            # extra_zeros = torch.ones_like(at) * 1e-8
            # extended_vt = torch.cat((vt, extra_zeros), dim=-1)
            # # [bz, d_vocab + lc]

            # wt = extended_vt.scatter_add(-1, index=c_ids_ext, src=at)
            # # [bz, d_vocab + lc]

            vocab_dist, attn_dist, context_vec, h, c = self.decoder(
                dec_input=ai, prev_h=h, prev_c=c, Y=Y, c_pad_mask=c_masks
            )

            # Eq. (8) - Compute generation probability p_gen
            context_feat = self.w_h(context_vec)  # [B x 1]
            decoder_feat = self.w_s(h)  # [B x 1]
            input_feat = self.w_x(ai)  # [B x 1]
            gen_feat = context_feat + decoder_feat + input_feat
            p_gen = torch.sigmoid(gen_feat)  # [B x 1]

            # Eq. (9) - Compute prob dist'n over extended vocabulary
            vocab_dist = p_gen * vocab_dist  # [B x V]
            weighted_attn_dist = (1.0 - p_gen) * attn_dist  # [B x L]

            # Concat some zeros to each vocab dist,
            # to hold probs for oov words that appeared in source text
            batch_size = vocab_dist.size(0)
            # extra_zeros = torch.zeros((batch_size, max_oov_len), device=vocab_dist.device)
            extra_zeros = torch.zeros((batch_size, max_oov_len), device=vocab_dist.device)
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)  # [B x V_x]

            final_dist = extended_vocab_dist.scatter_add(
                dim=-1, index=c_ids_ext, src=weighted_attn_dist
            )

            outputs.append(final_dist.unsqueeze(1))

        output_mle = torch.cat(outputs, 1)
        # [b, la, d_vocab + lc]

        return output_mle

    def do_predict(self, Y, q, q_masks, c_ids_ext, c_masks, max_oov_len):
        # Y: [bz, lc, 2 * d_hid]
        # q: [bz, lq, d_hid]
        # c_ids: [bz, lc]

        bz, _, d_hid = Y.size()
        d_hid = d_hid // 2

        # [bz, lc, d_hid]
        outputs = torch.zeros(bz, self.la, dtype=torch.long, device=Y.device)
        preds = torch.zeros(bz, self.la, dtype=torch.long, device=Y.device)
        h, c = torch.zeros(bz, d_hid, device=Y.device, dtype=Y.dtype), torch.zeros(
            bz, d_hid, device=Y.device, dtype=Y.dtype
        )
        for i in range(self.la):
            ## Embd token generated in last step
            last_tok = (
                outputs[0:1, i - 1 : i]
                if i > 0
                else torch.full(
                    (1, 1), self.sos_id, dtype=c_ids_ext.dtype, device=c_ids_ext.device
                )
            )
            # [1, 1]
            last_tok = self.embedding.embed_only(last_tok).squeeze(1)
            # [1, d_embd]
            # last_tok = self.lin1(last_tok)
            # # [1, d_hid]

            # ## Attention over Y
            # h_ = (
            #     self.ff1(h.transpose(0, 1).sum(dim=1)).unsqueeze(1)
            #     if torch.is_tensor(h)
            #     else 0
            # )
            # at = self.ff3(
            #     h_ + Y_ + self.attn_pooling(last_tok, q_, q_masks_).unsqueeze(1)
            # ).squeeze(-1)
            # # [1, lc]
            # ## Mask padding positions
            # at = at.float().masked_fill_(c_masks_ == 0, float("-inf")).type_as(q)
            # at = at.softmax(-1)
            # y = (at.unsqueeze(-1) * Y_).sum(dim=1)
            # # [1, d_hid]

            # ## Pass over LSTM
            # _, (h, c) = (
            #     self.lstm(torch.cat((y, last_tok), -1).unsqueeze(1), (h, c))
            #     if torch.is_tensor(h)
            #     else self.lstm(torch.cat((y, last_tok), -1).unsqueeze(1))
            # )
            # # h, c: [num_layers, 1, d_hid]

            # ## PGN
            # h_, c_ = h.transpose(0, 1).sum(dim=1), c.transpose(0, 1).sum(dim=1)
            # pt = F.sigmoid(self.lin_pgn1(c_) + self.lin_pgn2(h_) + self.lin_pgn3(y))
            # # [1, 1]

            # vt = self.lin_pgn(h_)
            # # [1, d_vocab]
            # vt = vt.softmax(-1)
            # vt = (1 - pt) * vt

            # at = pt * at

            # ## Padd zeros to vt for OOV positions
            # extra_zeros = torch.zeros_like(at)
            # extended_vt = torch.cat((vt, extra_zeros), dim=-1)
            # # [1, d_vocab + lc]

            # wt = extended_vt.scatter_add(-1, index=c_ids_ext_, src=at)
            # # [1, d_vocab + lc]

            vocab_dist, attn_dist, context_vec, h, c = self.decoder(
                dec_input=last_tok, prev_h=h, prev_c=c, Y=Y, c_pad_mask=c_masks
            )

            # Eq. (8) - Compute generation probability p_gen
            context_feat = self.w_h(context_vec)  # [B x 1]
            decoder_feat = self.w_s(h)  # [B x 1]
            input_feat = self.w_x(last_tok)  # [B x 1]
            gen_feat = context_feat + decoder_feat + input_feat
            p_gen = torch.sigmoid(gen_feat)  # [B x 1]

            # Eq. (9) - Compute prob dist'n over extended vocabulary
            vocab_dist = p_gen * vocab_dist  # [B x V]
            weighted_attn_dist = (1.0 - p_gen) * attn_dist  # [B x L]

            # Concat some zeros to each vocab dist,
            # to hold probs for oov words that appeared in source text
            batch_size = vocab_dist.size(0)
            extra_zeros = torch.zeros((batch_size, max_oov_len), device=vocab_dist.device)
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)  # [B x V_x]

            final_dist = extended_vocab_dist.scatter_add(
                dim=-1, index=c_ids_ext, src=weighted_attn_dist
            )

            pred_tok = final_dist.argmax(-1)
            # [1]
            preds[0, i] = pred_tok
            outputs[0, i] = (
                pred_tok
                if pred_tok < self.d_vocab
                else torch.tensor([self.vocab.unk()], device=pred_tok.device, dtype=pred_tok.dtype)
            )
            if pred_tok == self.eos_id:
                break

        return preds
