from re import A

import torch
import torch.nn as torch_nn
from utils.model_utils import ipot

from models.layers import PGN, Embedding, IntrospectiveAlignmentLayer


class Backbone(torch_nn.Module):
    def __init__(
        self,
        batch_size,
        la,
        lc,
        nc,
        d_bert,
        d_hid,
        d_vocab,
        num_layers_lstm,
        block,
        dropout,
        path_pretrained,
        criterion,
        device,
    ):
        super().__init__()

        self.d_vocab = d_vocab

        self.embedding = Embedding(d_hid, d_bert, path_pretrained)
        self.ial = IntrospectiveAlignmentLayer(batch_size, nc, lc, d_hid, dropout, block, device)
        self.pgn = PGN(
            la, d_hid, d_vocab, num_layers_lstm, dropout, path_pretrained, self.embedding
        )

        self.criterion = criterion
    
    def _ids2dist(self, inputs):
        indices = inputs.unsqueeze(-1)
        a = torch.full((*inputs.size(), self.d_vocab), 1e-6, dtype=torch.float, device=inputs.device)
        a.scatter_(
            dim=-1,
            index=indices,
            src=torch.full(indices.size(), 0.99, dtype=torch.float, device=inputs.device),
        )
        return a

    def get_loss(self, output_mle, trgs, is_loss_ot=False, gamma=0.08):
        loss = 0
        for output, trg in zip(output_mle, trgs):
            loss_mle = self.criterion(output, trg)

            if is_loss_ot:
                trg = self.encoder(trg)
                pred = (
                    torch.softmax(output_mle.transpose(-1, -2), dim=-1)
                    @ self.encoder.embeddings.word_embeddings.weight
                )
                loss_ot = ipot(pred, trg, max_iter=400)
            else:
                loss_ot = 0

            loss += loss_mle + gamma * loss_ot

        return loss / len(trgs)

    def get_trgs(self, a_ids):
        # [b, l_]
        return torch.cat(
            (
                a_ids[:, 1:],
                torch.zeros((a_ids.size(0), 1), dtype=a_ids.dtype, device=a_ids.device),
            ),
            -1,
        )

    def do_train(self, q_ids, c_ids, a1_ids, a2_ids, use_2_answers=False):
        # q_ids: [b, lq]
        # c_ids: [b, nc, lc]
        # a1_ids: [b, la]
        # a2_ids: [b, la]

        # l = nc * lc

        q, c, c_ids = self.embedding.embed_qc(q_ids, c_ids)
        # q, c: [bz, l_, d_hid]
        # c_ids: [bz, nc*lc]

        Y = self.ial(q, c)
        # [bz, l, 2*d_hid]

        output_mle, trgs = [], []
        ans = [a1_ids, a2_ids] if use_2_answers else [a1_ids]
        for a_ids in ans:
            a = self.embedding.embed_a(a_ids)
            # a: [bz, l_, d_hid]

            output_mle_ = self.pgn.do_train(Y, q, a, c_ids)
            trg = self.get_trgs(a_ids)

            output_mle.append(output_mle_)
            trgs.append(trg)

        loss = self.get_loss(output_mle, trgs)

        return loss, output_mle

    def do_predict(self, q_ids, c_ids, a1_ids, a2_ids, use_2_answers=False):
        # q_ids: [b, lq]
        # c_ids: [b, nc, lc]
        # a1_ids: [b, la]
        # a2_ids: [b, la]

        q, c, c_ids = self.embedding.embed_qc(q_ids, c_ids)
        # q, c: [bz, l_, d_hid]
        # c_ids: [bz, nc*lc]

        Y = self.ial(q, c)
        # [bz, l, 2*d_hid]

        outputs = self.pgn.do_predict(Y, q, c_ids)

        output_mle_ = self._ids2dist(outputs).transpose(-1, -2)
        # [b, la, d_vocab]

        if use_2_answers:
            trgs = [self.get_trgs(a_ids) for a_ids in [a1_ids, a2_ids]]
            output_mle = [output_mle_, output_mle_]
        else:
            trgs = [self.get_trgs(a1_ids)]
            output_mle = [output_mle_]
        loss = self.get_loss(output_mle, trgs)

        return loss, outputs
