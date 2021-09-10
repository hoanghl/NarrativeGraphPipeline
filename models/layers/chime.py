import torch
import torch.nn as torch_nn
from models.layers.graph_layer import GraphBasedReasoningLayer
from transformers import BertModel, BertTokenizer, logging
from utils.model_utils import TransEncoder

logging.set_verbosity_error()

import itertools
import random


class CHIME(torch_nn.Module):
    def __init__(
        self,
        lq,
        lc,
        d_bert,
        d_hid,
        d_graph,
        n_edges,
        d_vocab,
        dropout,
        n_propagations,
        path_pretrained,
        criterion,
    ):
        super().__init__()

        self.lq = lq
        self.lc = lc
        self.la = -1
        tokenizer = BertTokenizer.from_pretrained(path_pretrained)
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.n_edges = n_edges

        self.encoder = BertModel.from_pretrained(path_pretrained)
        self.graph = GraphBasedReasoningLayer(
            d_hid, d_bert, d_graph, n_propagations, n_edges, dropout
        )
        self.lin1 = torch_nn.Linear(d_bert * 2, d_bert, bias=False)
        self.c_mem_encoder = TransEncoder(num_layers=1, model_dim=d_bert, num_heads=8)
        self.a_mem_encoder = TransEncoder(num_layers=1, model_dim=d_bert, num_heads=8)
        self.gate_c = torch_nn.Sequential(torch_nn.Linear(d_bert * 2, 1), torch_nn.Sigmoid())
        self.gate_a = torch_nn.Sequential(torch_nn.Linear(d_bert * 2, 1), torch_nn.Sigmoid())
        self.decoder = torch_nn.Linear(d_bert, d_vocab)

        self.criterion = criterion
        ## Init
        self.decoder.weight = self.encoder.embeddings.word_embeddings.weight
        self.decoder.bias.data.zero_()

    # np: no pad token
    # lq_np : seq len of question (no pad)
    # lc_np : seq len of context (no pad)
    # la_np : seq len of answer (no pad)
    # l_qc = 1 + lq + 1 + lc + 1
    # l_qca = 1 + lq + 1 + lc + 1 + la + 1

    def _get_edge_index(self, c_masks):
        def gen_edges(c_mask):
            # c_masks: [nc, lc]
            nc = c_mask.size(0)
            n_nodes = c_mask[:, 0].sum()

            edges = list(itertools.combinations(range(n_nodes), 2))
            random.shuffle(edges)

            vertex_s, vertex_d = [], []
            for edge in edges[: self.n_edges // 2]:
                s, d = edge

                vertex_s.append(int(s))
                vertex_d.append(int(d))

                vertex_s.append(int(d))
                vertex_d.append(int(s))

            # Pad edge indices
            vertex_s += [nc - 1] * (self.n_edges - len(vertex_s))
            vertex_d += [nc - 1] * (self.n_edges - len(vertex_d))

            edge_index = torch.tensor(
                [vertex_s, vertex_d], dtype=torch.long, device=c_masks.device
            )
            # [2, n_edges]

            return edge_index

        edge_indx = []
        for c_mask in c_masks:
            edge_indx_ = gen_edges(c_mask)
            edge_indx.append(edge_indx_.unsqueeze(0))

        edge_indx = torch.cat(edge_indx, dim=0).type_as(c_masks)

        return edge_indx

    def _get_loss(self, output_mle, trgs):
        return self.criterion(output_mle.transpose(-1, -2), trgs)

    def _get_padding_mask(self, qca_ids, sen_1_leng):
        s_l = qca_ids.size(0)
        ones = torch.ones((s_l, s_l), device=qca_ids.device)
        mask = ones.tril()
        mask[:, :sen_1_leng] = 1
        padding_mask = (qca_ids != self.pad_id).float()
        mask *= padding_mask
        mask *= padding_mask.unsqueeze(1)
        return mask

    def create_misc(self, q, c, a):
        # q: [b, lq]
        # r: [b, lc]
        # a: [b, la]

        bz = q.size(0)
        device = q.device

        ## Create mask and tok_type_id
        qca_ids = torch.zeros(
            bz,
            1 + q.size(1) + 1 + c.size(1) + 1 + a.size(1) + 1,
            dtype=torch.long,
            device=q.device,
        )
        # NOTE: If not work, use the next of following instead of the following
        # qca_masks = torch.zeros_like(qca_ids)
        qca_masks = torch.ones((bz, qca_ids.size(1), qca_ids.size(1)), device=device)
        qca_tok_type_id = torch.ones_like(qca_ids)
        trgs = []
        lq_np, lc_np, la_np = [], [], []
        for i in range(bz):
            q_, c_, a_ = q[i], c[i], a[i]

            q_np = q_[q_ != self.pad_id]
            c_np = c_[c_ != self.pad_id]
            a_np = a_[a_ != self.pad_id]
            lq_ = len(q_np)
            lc_ = len(c_np)
            la_ = len(a_np)
            lq_np.append(len(q_np))
            lc_np.append(len(c_np))
            la_np.append(len(a_np))

            l = 1 + lq_ + 1 + lc_ + 1 + la_ + 1
            qca_ids[i, :l] = torch.cat(
                [
                    torch.tensor([self.cls_id], device=device, dtype=torch.long),
                    q_np,
                    # NOTE: If not work, use the next of following instead of the following
                    torch.tensor([self.sep_id], device=device, dtype=torch.long),
                    # torch.tensor([self.cls_id], device=device, dtype=torch.long),
                    c_np,
                    torch.tensor([self.sep_id], device=device, dtype=torch.long),
                    a_np,
                    torch.tensor([self.sep_id], device=device, dtype=torch.long),
                ],
                dim=-1,
            )
            l = 1 + lq_ + 1 + lc_ + 1
            # NOTE: If not work, use the next of following instead of the following
            # qca_masks[i, :l] = torch.ones((l,), dtype=torch.long, device=device)
            qca_masks[i] = self._get_padding_mask(qca_ids[i], l)

            qca_tok_type_id[i, : 1 + lq_ + 1] = torch.zeros(
                (1 + lq_ + 1,), dtype=torch.long, device=device
            )
            l = 1 + lq_ + 1 + lc_ + 1
            qca_tok_type_id[i, l : l + la_ + 1] = torch.zeros(
                (la_ + 1,), dtype=torch.long, device=device
            )

            trg = torch.cat(
                [
                    a_np,
                    torch.full((1,), self.sep_id, dtype=torch.long, device=device),
                    torch.full((self.la - la_ + 1,), self.pad_id, dtype=torch.long, device=device),
                ],
                dim=-1,
            )
            trgs.append(trg.unsqueeze(0))

        trgs = torch.cat(trgs, dim=0)

        return qca_ids, qca_masks, qca_tok_type_id, trgs, lq_np, lc_np, la_np

    def encode(self, qca_ids, qca_masks, qca_tok_type_id, lq_np, lc_np, la_np):
        outputs = self.encoder(qca_ids, qca_masks, qca_tok_type_id)[0]

        part1, part2, c_hid = [], [], []
        for output, lq_, lc_, la_ in zip(outputs, lq_np, lc_np, la_np):
            l_pad_q = self.lq - lq_
            l_pad_c = self.lc - lc_

            l1 = 1 + lq_ + 1
            l2 = 1 + lq_ + 1 + lc_ + 1 + la_ + 1 + l_pad_q
            c_ = torch.cat(
                [output[l1 : l1 + lc_], output[l2 : l2 + l_pad_c]],
                dim=0,
            )

            l1 = 1 + lq_ + 1
            l2 = 1 + lq_ + 1 + lc_ + 1 + la_ + 1
            p1 = torch.cat(
                [output[1 : 1 + lq_], output[l1 : l1 + lc_], output[l2 : l2 + l_pad_q + l_pad_c]],
                dim=0,
            )
            l1 = 1 + lq_ + 1 + lc_
            l2 = 1 + lq_ + 1 + lc_ + 1 + la_ + 1 + l_pad_q + l_pad_c
            p2 = torch.cat([output[l1 : l1 + 1 + la_ + 1], output[l2:]], dim=0)

            c_hid.append(c_.unsqueeze(0))
            part1.append(p1.unsqueeze(0))
            part2.append(p2.unsqueeze(0))

        return torch.cat(part1, dim=0), torch.cat(part2, dim=0), torch.cat(c_hid, dim=0)

    def forward(self, q, c, a, edge_index):
        # q: [b, lq]
        # c: [b, nc, lc]
        # a: [b, la]
        # edge_index: [b, 2, n_edges]

        _, nc, lc = c.size()

        self.la = a.size(-1)

        ####################
        # Encode qca using Bert
        ####################
        part1, part2, c_hid = [], [], []
        for n in range(nc):
            # From q, c and a, create id, mask and tok_type_id tensors
            qca_ids, qca_masks, qca_tok_type_id, trg, lq_np, lc_np, la_np = self.create_misc(
                q, c[:, n], a
            )

            # Encode qca using Bert
            p1, p2, c_hid_ = self.encode(qca_ids, qca_masks, qca_tok_type_id, lq_np, lc_np, la_np)

            part1.append(p1)
            part2.append(p2)
            c_hid.append(c_hid_.unsqueeze(1))
        c_hid = torch.cat(c_hid, dim=1)
        # [b, nc, lc, d_bert]

        ####################
        # Apply GCN
        ####################
        c_hid = self.graph(c_hid, edge_index)
        # [b, nc, d_bert]
        c_hid = c_hid.unsqueeze(2).repeat(1, 1, self.lc + self.lq, 1)
        # [b, nc, lc, d_bert]

        ####################
        # Apply TransEncoder with memory
        ####################
        c_mem, a_mem = None, None
        for n, (
            p1,
            p2,
        ) in enumerate(zip(part1, part2)):

            x = torch.cat((c_hid[:, n], p1), dim=-1)
            # [b, lc, d_bert * 2]
            p1 = self.lin1(x)
            # [b, lc, d_bert]

            if not torch.is_tensor(c_mem):
                c_mem, a_mem = p1, p2
            else:
                ## Apply TransEncoder with p1
                z = self.c_mem_encoder(c_mem, p1, p1)
                c_gate = self.gate_c(torch.cat((z, c_mem), dim=-1))
                c_mem = c_gate * c_mem + (1 - c_gate) * p1

                ## Apply TransEncoder with p2
                z = self.a_mem_encoder(a_mem, c_mem, c_mem)
                a_gate = self.gate_a(torch.cat((z, a_mem), dim=-1))
                a_mem = a_gate * a_mem + (1 - a_gate) * p2

        output_mle = self.decoder(a_mem)
        # [b, la + 2, d_vocab]

        return output_mle, trg

    def do_train(self, q, c, a1, a2, c_masks, use_2_answers=False):
        edge_index = self._get_edge_index(c_masks)
        # [b, 2, n_edges]

        output_mle, trgs = self(q, c, a1, edge_index)

        loss = self._get_loss(output_mle, trgs)

        return loss, output_mle

    def do_predict(self, q, c, c_masks, la):
        bz = q.size(0)

        edge_index = self._get_edge_index(c_masks)
        # [b, 2, n_edges]

        ans = []
        for i in range(bz):
            q_, c_ = q[i : i + 1], c[i : i + 1]
            a = torch.tensor([[]], device=q.device)

            for _ in range(la):
                output, _ = self(q_, c_, a, edge_index)
                topi = torch.log_softmax(output[:, :, a.size(-1)], dim=-1).argmax(dim=-1)
                if topi == self.sep_id:
                    break
                a = torch.cat((a, topi.unsqueeze(-1)), dim=-1)

            a_ = torch.full((1, la), self.pad_id, dtype=a.dtype, device=a.device)
            a_[0, : a.size(-1)] = a[0]
            ans.append(a_)

        return torch.cat(ans, dim=0)
