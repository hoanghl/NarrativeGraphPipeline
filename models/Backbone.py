from re import A

import torch
import torch.nn as nn
from utils.model_utils import ipot

from models.layers import PGN, Embedding, IntrospectiveAlignmentLayer


class Backbone(nn.Module):
    def __init__(
        self,
        batch_size,
        la,
        lc,
        nc,
        d_embd,
        d_hid,
        d_vocab,
        num_layers,
        block,
        dropout,
        path_pretrained,
        path_vocab,
        criterion,
        device,
    ):
        super().__init__()

        self.d_vocab = d_vocab
        self.lc = lc

        self.embedding = Embedding(d_embd, d_hid, path_pretrained, path_vocab, num_layers, dropout)
        self.ial = IntrospectiveAlignmentLayer(batch_size, lc, d_hid, dropout, block, device)
        self.pgn = PGN(la, d_embd, d_hid, d_vocab, num_layers, dropout, path_vocab, self.embedding)

        self.criterion = criterion

    def _ids2dist(self, inputs):
        indices = inputs.unsqueeze(-1)
        a = torch.full(
            (*inputs.size(), self.d_vocab + self.lc), 1e-6, dtype=torch.float, device=inputs.device
        )
        a.scatter_(
            dim=-1,
            index=indices,
            src=torch.full(indices.size(), 0.99, dtype=torch.float, device=inputs.device),
        )
        return a

    def _get_loss(self, output_mle, trgs, is_loss_ot=False, gamma=0.08):
        # output_mle: list of [bz, la, d_vocab + lc]
        # trgs: list of [bz, la]

        loss = 0
        for output, trg in zip(output_mle, trgs):
            loss_mle = self.criterion(output.transpose(-1, -2), trg)

            if is_loss_ot:
                trg = self.encoder(trg)
                pred = (
                    torch.softmax(output_mle, dim=-1)
                    @ self.encoder.embeddings.word_embeddings.weight
                )
                loss_ot = ipot(pred, trg, max_iter=400)
            else:
                loss_ot = 0

            loss += loss_mle + gamma * loss_ot

        return loss / len(trgs)

    def do_train(
        self,
        q_ids,
        q_masks,
        c_ids,
        c_ids_ext,
        c_masks,
        dec_inp1_ids,
        dec_inp2_ids,
        dec_trg1_ids,
        dec_trg2_ids,
        use_2_answers=False,
        **kwargs
    ):
        # q_ids: [bz, lq]
        # q_masks: [bz, lq]
        # c_ids: [bz, lc]
        # c_masks: [bz, lc]
        # c_ids_ext: [bz, lc]
        # dec_inp1_ids: [bz, la]
        # dec_inp2_ids: [bz, la]
        # dec_trg1_ids: [bz, la]
        # dec_trg2_ids: [bz, la]

        q, c = self.embedding(q_ids, q_masks), self.embedding(c_ids, c_masks)
        # q, c: [bz, l_, d_hid]

        Y = self.ial(q, c)
        # [bz, l, 2*d_hid]

        output_mle = []
        ans = [dec_inp1_ids, dec_inp2_ids] if use_2_answers else [dec_inp1_ids]
        trgs = [dec_trg1_ids, dec_trg2_ids] if use_2_answers else [dec_trg1_ids]
        for a_ids in ans:
            a = self.embedding.embed_only(a_ids)
            # [bz, la, d_embd]

            output_mle_ = self.pgn.do_train(Y, q, q_masks, a, c_ids_ext, c_masks)
            # [bz, la, d_vocab + lc]

            output_mle.append(output_mle_)

        loss = self._get_loss(output_mle, trgs)

        return loss, output_mle

    def do_predict(
        self,
        q_ids,
        q_masks,
        c_ids,
        c_ids_ext,
        c_masks,
        dec_trg1_ids,
        dec_trg2_ids,
        use_2_answers=False,
        **kwargs
    ):
        # q_ids: [bz, lq]
        # q_masks: [bz, lq]
        # c_ids: [bz, lc]
        # c_masks: [bz, lc]
        # c_ids_ext: [bz, lc]
        # dec_inp1_ids: [bz, la]
        # dec_inp2_ids: [bz, la]
        # dec_trg1_ids: [bz, la]
        # dec_trg2_ids: [bz, la]

        q, c = self.embedding(q_ids, q_masks), self.embedding(c_ids, c_masks)
        # q, c: [bz, l_, d_hid]

        Y = self.ial(q, c)
        # [bz, l, 2*d_hid]

        outputs = self.pgn.do_predict(Y, q, q_masks, c_ids_ext, c_masks)
        # [bz]

        output_mle_ = self._ids2dist(outputs)
        # [b, la, d_vocab + lc]

        if use_2_answers:
            trgs = [dec_trg1_ids, dec_trg2_ids]
            output_mle = [output_mle_, output_mle_]
        else:
            trgs = [dec_trg1_ids]
            output_mle = [output_mle_]

        loss = self._get_loss(output_mle, trgs)

        return loss, outputs
