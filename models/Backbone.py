import torch
import torch.nn as nn
from utils.model_utils import Loss

from models.layers import PGN, Encoder, IntrospectiveAlignmentLayer


class Backbone(nn.Module):
    def __init__(
        self,
        la,
        lc,
        d_embd,
        d_hid,
        d_vocab,
        num_layers,
        block,
        dropout,
        path_pretrained,
        vocab,
    ):
        super().__init__()

        self.d_vocab = d_vocab
        self.lc = lc

        self.encoder = Encoder(d_embd, d_hid, path_pretrained, num_layers, dropout, vocab)
        self.ial = IntrospectiveAlignmentLayer(d_hid * 2, dropout, block)
        self.pgn = PGN(la, d_embd, d_hid * 2, d_vocab, dropout, self.encoder, vocab)

        self.criterion = Loss(use_coverage=True, use_ot_loss=False, encoder=self.encoder)

    def _ids2dist(self, inputs, max_oov_len):
        indices = inputs.unsqueeze(-1)
        a = torch.full(
            (*inputs.size(), self.d_vocab + max_oov_len),
            1e-6,
            dtype=torch.float,
            device=inputs.device,
        )
        a.scatter_(
            dim=-1,
            index=indices,
            src=torch.full(indices.size(), 0.99, dtype=torch.float, device=inputs.device),
        )
        return a

    def do_train(self, batch, use_2_answers=False):
        q, c = self.encoder(batch.q_input, batch.q_len), self.encoder(batch.c_input, batch.c_len)
        # q, c: [bz, l_, d_hid*2]

        Y = self.ial(q, c)
        # [bz, l, d_hid*2]

        if use_2_answers:
            # FIXME: Fix this using second answer
            ans = [batch.dec_input, batch.dec_input]
            trgs = [batch.dec_target, batch.dec_target]
            dec_masks = [batch.dec_pad_mask, batch.dec_pad_mask]
            dec_lens = [batch.dec_len, batch.dec_len]
        else:
            ans = [batch.dec_input]
            trgs = [batch.dec_target]
            dec_masks = [batch.dec_pad_mask]
            dec_lens = [batch.dec_len]

        output_mle = []
        loss = 0
        for a_ids, trg, dec_mask, dec_len in zip(ans, trgs, dec_masks, dec_lens):
            a = self.encoder.embed_only(a_ids)
            # [bz, la, d_embd]

            output_mle_, cov, att_dist = self.pgn.do_train(
                Y, q, batch.q_pad_mask, a, batch.c_input_ext, batch.c_pad_mask, batch.max_oov_len
            )
            # output_mle_: [bz, d_vocab + max_oov_len, la]
            # cov: [bz, lc, la]
            # att_dist: [bz, lc, la]

            loss = loss + self.criterion(output_mle_, cov, att_dist, trg, dec_mask, dec_len)

            output_mle.append(output_mle_)

        loss = loss / len(output_mle)

        return loss, output_mle

    def do_predict(self, batch, use_2_answers=False):
        q, c = self.encoder(batch.q_input, batch.q_len), self.encoder(batch.c_input, batch.c_len)
        # q, c: [bz, l_, d_hid*2]

        Y = self.ial(q, c)
        # [bz, l, d_hid*2]

        preds, output_mle, coverage, attn_dists = self.pgn.do_predict(
            Y, q, batch.q_pad_mask, batch.c_input_ext, batch.c_pad_mask, batch.max_oov_len
        )

        if use_2_answers:
            # FIXME: Fix this using second answer
            trgs = [batch.dec_target, batch.dec_target]
            dec_masks = [batch.dec_pad_mask, batch.dec_pad_mask]
            dec_lens = [batch.dec_len, batch.dec_len]
        else:
            trgs = [batch.dec_target]
            dec_masks = [batch.dec_pad_mask]
            dec_lens = [batch.dec_len]
        loss = 0
        for trg, dec_mask, dec_len in zip(trgs, dec_masks, dec_lens):
            loss = loss + self.criterion(output_mle, coverage, attn_dists, trg, dec_mask, dec_len)

        return loss, preds
