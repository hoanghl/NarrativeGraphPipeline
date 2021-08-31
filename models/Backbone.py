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

        self.embedding = Embedding(d_hid, d_bert, path_pretrained)
        self.ial = IntrospectiveAlignmentLayer(batch_size, nc, lc, d_hid, dropout, block, device)
        self.pgn = PGN(
            la, d_hid, d_vocab, num_layers_lstm, dropout, path_pretrained, self.embedding
        )

        self.criterion = criterion

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

    def do_train(self, q_ids, c_ids, a1_ids, a2_ids):
        # q_ids: [b, lq]
        # c_ids: [b, nc, lc]
        # a1_ids: [b, la]
        # a2_ids: [b, la]

        # l = nc * lc

        q, c, c_ids = self.embedding.embed_qc(q_ids, c_ids)
        a1, a2 = self.embedding.embed_a(a1_ids), self.embedding.embed_a(a2_ids)
        # q, c, a1, a2: [bz, l_, d_hid]
        # c_ids: [bz, nc*lc]

        Y = self.ial(q, c)
        # [bz, l, 2*d_hid]

        output_mle = [self.pgn.do_train(Y, q, a, c_ids) for a in [a1, a2]]
        trgs = [self.get_trgs(a_ids) for a_ids in [a1_ids, a2_ids]]

        loss = self.get_loss(output_mle, trgs)

        return loss, output_mle

    def do_predict(self, q_ids, c_ids, a1_ids, a2_ids):
        # q_ids: [b, lq]
        # c_ids: [b, nc, lc]
        # a1_ids: [b, la]
        # a2_ids: [b, la]

        q, c, c_ids = self.embedding.embed_qc(q_ids, c_ids)
        # q, c: [bz, l_, d_hid]
        # c_ids: [bz, nc*lc]

        Y = self.ial(q, c)
        # [bz, l, 2*d_hid]

        output_mle, outputs = self.pgn.do_predict(Y, q, c_ids)
        trgs = [self.get_trgs(a_ids) for a_ids in [a1_ids, a2_ids]]

        loss = self.get_loss([output_mle, output_mle], trgs)

        return loss, outputs
