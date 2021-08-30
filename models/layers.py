import torch
import torch.nn as torch_nn
from torch.nn.parameter import Parameter
from transformers import BertModel, BertTokenizer, logging
from utils.model_utils import EncoderLayer, TransEncoder


class Embedding(torch_nn.Module):
    def __init__(self, lq, lc, d_bert, num_heads, path_pretrained, device):
        super().__init__()

        tokenizer = BertTokenizer.from_pretrained(path_pretrained)
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.lq = lq
        self.lc = lc

        self.encoder = BertModel.from_pretrained(path_pretrained)
        # NOTE: Temporarily comment this Persistent module
        # self.per_mem = PersistentMemoryCell(
        #     lq=lq, lc=lc, num_heads=num_heads, d_bert=d_bert, device=device
        # )

    def _get_padding_mask(self, qca_ids, sen_1_leng):
        s_l = qca_ids.size(0)
        ones = torch.ones((s_l, s_l), device=qca_ids.device)
        mask = ones.tril()
        mask[:, :sen_1_leng] = 1
        padding_mask = (qca_ids != self.pad_id).float()
        mask *= padding_mask
        mask *= padding_mask.unsqueeze(1)
        return mask

    def create_misc(self, q, c, a, la):
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
                    torch.full((la - la_ + 1,), self.pad_id, dtype=torch.long, device=device),
                ],
                dim=-1,
            )
            trgs.append(trg.unsqueeze(0))

        trgs = torch.cat(trgs, dim=0)

        return qca_ids, qca_masks, qca_tok_type_id, trgs, lq_np, lc_np, la_np

    def encode(self, qca_ids, qca_masks, qca_tok_type_id, lq_np, lc_np, la_np):
        outputs = self.encoder(qca_ids, qca_masks, qca_tok_type_id)[0]

        q_parts, c_parts, a_parts = [], [], []
        for output, lq_, lc_, la_ in zip(outputs, lq_np, lc_np, la_np):
            l_pad_q = self.lq - lq_
            l_pad_c = self.lc - lc_

            ## Get c part in qca and its padding
            l1 = 1 + lq_ + 1
            l2 = 1 + lq_ + 1 + lc_ + 1 + la_ + 1 + l_pad_q
            c_part = torch.cat(
                [output[l1 : l1 + lc_], output[l2 : l2 + l_pad_c]],
                dim=0,
            )

            ## Get q part in qca and its padding
            l1 = 1
            l2 = 1 + lq_ + 1 + lc_ + 1 + la_ + 1
            q_part = torch.cat(
                [output[l1 : l1 + lq_], output[l2 : l2 + l_pad_q]],
                dim=0,
            )

            ## Get a part in qca and its padding
            l1 = 1 + lq_ + 1 + lc_
            l2 = 1 + lq_ + 1 + lc_ + 1 + la_ + 1 + l_pad_q + l_pad_c
            a_part = torch.cat([output[l1 : l1 + 1 + la_ + 1], output[l2:]], dim=0)

            ## Apply Persistent Memory filter and apply back to c_part
            # NOTE: Temporarily comment this Persistent module
            # c_filter = self.per_mem(q_part.unsqueeze(0), c_part.unsqueeze(0))
            # # [1, lc - 2]
            # c_part = c_part * c_filter.view(c_filter.size(-1), 1)
            # # [lc - 2, d_bert]

            q_parts.append(q_part.unsqueeze(0))
            c_parts.append(c_part.unsqueeze(0))
            a_parts.append(a_part.unsqueeze(0))

        return torch.cat(q_parts, dim=0), torch.cat(c_parts, dim=0), torch.cat(a_parts, dim=0)

    def forward(self, q, ci, a, la):
        # Create parts
        qca_ids, qca_masks, qca_tok_type_id, trgs, lq_np, lc_np, la_np = self.create_misc(
            q, ci, a, la
        )

        # Encode qca using Bert
        q, ci, a = self.encode(qca_ids, qca_masks, qca_tok_type_id, lq_np, lc_np, la_np)

        return q, ci, a, trgs


class DualAttention(torch_nn.Module):
    def __init__(self, d_hid):
        super().__init__()

        self.lin_dual1 = torch_nn.Linear(d_hid * 3, 1)

    def forward(self, q, ci):
        # q: [b, lq, d]
        # ci: [b, lc, d]

        lc, lq = ci.size(1), q.size(1)

        U = []
        for j in range(lq):
            q_ = q[:, j : j + 1]
            tmp = torch.cat((ci, q_.repeat(1, lc, 1), ci * q_), -1)
            # [b, lc, d * 3]
            tmp = self.lin_dual1(tmp)
            # [b, lc, 1]
            U.append(tmp)

        U = torch.cat(U, -1)
        # [b, lc, lq]

        Ak = torch.softmax(U, -1)
        Bk = torch.softmax(U.transpose(-1, -2), -1)
        # [b, lq, lc]
        Ak1 = Ak @ q
        # [b, lc, d]
        Bk1 = Bk @ ci
        # [b, lq, d]
        Ak2 = Ak @ Bk1
        # [b, lc, d]
        Bk2 = Bk @ Ak1
        # [b, lq, d]

        G_qp = torch.cat((ci, Ak1, Ak2, ci * Ak1, ci * Ak2), -1)
        # [b, lc, 5d]

        G_pq = torch.cat((q, Bk1, Bk2, q * Bk1, q * Bk2), -1)
        # [b, lq, 5d]

        return G_pq, G_qp


class ModelingEncoder(torch_nn.Module):
    def __init__(self, num_layers_p, num_layers_q, num_heads, d_hid, dropout):
        super().__init__()

        self.lin = torch_nn.Linear(d_hid * 5, d_hid)
        self.enc_p = TransEncoder(
            num_layers=num_layers_p,
            model_dim=d_hid,
            num_heads=num_heads,
            no_pos_embd=True,
            dropout=dropout,
        )
        self.enc_q = TransEncoder(
            num_layers=num_layers_q,
            model_dim=d_hid,
            num_heads=num_heads,
            no_pos_embd=True,
            dropout=dropout,
        )

    def forward(self, G_pq, G_qp):
        # G_pq: [b, lq, 5d]
        # G_qp: [b, lc, 5d]

        G_pq, G_qp = self.lin(G_pq), self.lin(G_qp)
        # G_pq: [b, lq, d]
        # G_qp: [b, lc, d]

        Mq, Mp = self.enc_q(G_pq, G_pq, G_pq), self.enc_p(G_qp, G_qp, G_qp)
        # Mq: [b, lq, d]
        # Mp: [b, lc, d]

        return Mq, Mp


class AnsEncoder(torch_nn.Module):
    def __init__(self, d, num_layers, num_heads, dropout):
        super().__init__()

        self.encoder_q = EncoderLayer(num_heads=num_heads, model_dim=d, no_pos_embd=True)
        self.encoder_p = EncoderLayer(num_heads=num_heads, model_dim=d, no_pos_embd=True)
        self.ff = torch_nn.Sequential(
            torch_nn.Linear(d, d), torch_nn.Tanh(), torch_nn.Dropout(dropout)
        )
        self.norm = torch_nn.LayerNorm(d)
        self.num_layers = num_layers

    def forward(self, a, Mq, Mp):
        # a: [b, la, d]
        # Mq: [b, lq, d]
        # Mp: [b, lc, d]

        for _ in range(self.num_layers):
            output, _ = self.encoder_q(a, Mq, Mq)
            # [b, la, d]

            output, _ = self.encoder_p(output, Mp, Mp)
            # [b, la, d]

            a = self.norm(self.ff(output) + output)
            # [b, la, d]

        return a


class ShortTermMemoryCell(torch_nn.Module):
    def __init__(self, d, dropout):
        super().__init__()

        self.lin1 = torch_nn.Linear(d * 2, d, bias=False)
        self.ff = torch_nn.Sequential(
            torch_nn.Linear(d * 3, d), torch_nn.Tanh(), torch_nn.Dropout(dropout)
        )
        self.q_mem_enc = TransEncoder(num_layers=1, model_dim=d, num_heads=8)
        self.c_mem_enc = TransEncoder(num_layers=1, model_dim=d, num_heads=8)
        self.a_mem_enc = TransEncoder(num_layers=1, model_dim=d, num_heads=8)
        self.gate_q = torch_nn.Sequential(torch_nn.Linear(d * 2, 1), torch_nn.Sigmoid())
        self.gate_c = torch_nn.Sequential(torch_nn.Linear(d * 2, 1), torch_nn.Sigmoid())
        self.gate_a = torch_nn.Sequential(torch_nn.Linear(d * 2, 1), torch_nn.Sigmoid())

    def forward(self, q, c, a, q_mem, c_mem, a_mem):
        if not torch.is_tensor(q_mem):
            return q, c, a

        ## Use TransEnc to retrieve from new tensors q and c
        q_ = self.q_mem_enc(q_mem, q, q)
        # [b, lq, d]
        c_ = self.q_mem_enc(c_mem, c, c)
        # [b, lc, d]

        q_retrv = self.a_mem_enc(a, q, q)
        # [b, la, d]
        c_retrv = self.a_mem_enc(a, c, c)
        # [b, la, d]
        a_retrv = self.ff(torch.cat((q_retrv, c_retrv, a), -1))
        # [b, la, d]

        ## Update memory
        gate_q = self.gate_q(torch.cat((q_, q), -1))
        # [b, lq, 1]
        gate_c = self.gate_c(torch.cat((c_, c), -1))
        # [b, lc, 1]
        gate_a = self.gate_a(torch.cat((a_retrv, a), -1))
        # [b, la, 1]
        q_mem = gate_q * q_mem + (1 - gate_q) * q
        c_mem = gate_c * c_mem + (1 - gate_c) * c
        a_mem = gate_a * a_mem + (1 - gate_a) * a

        return q_mem, c_mem, a_mem


class SelfAttnQueryRetrv(torch_nn.Module):
    def __init__(self, lq, lc, num_heads, d_bert, device):
        super().__init__()

        self.lc = lc
        self.lq = lq

        d_k = d_bert // num_heads
        self.W_q = Parameter(torch.rand(num_heads, d_bert, d_k, device=device))
        self.W_c = Parameter(torch.rand(num_heads, d_bert, d_k, device=device))
        self.W_m = Parameter(torch.rand(num_heads, lc, lc, device=device))
        self.W_0 = Parameter(torch.rand(num_heads, 1, device=device))
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.lc, self.lc),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(lq),
        )

    def forward(self, q, c, m):
        # q: [b, lq, d_bert]
        # c: [b, lc, d_bert]
        # m: [lc, lc]

        b = c.size(0)

        q_proj = q.unsqueeze(1) @ self.W_q
        # [b, num_heads, lq, d_k]
        c_proj = c.unsqueeze(1) @ self.W_c
        # [b, num_heads, lc, d_k]
        m_proj = m.view(1, 1, self.lc, self.lc) @ self.W_m
        # [b, num_heads, lc, lc]

        product = q_proj @ c_proj.transpose(-1, -2)
        # [b, num_heads, lq, lc]
        product = self.ff1(product.view(-1, self.lq, self.lc)).view(b, -1, self.lq, self.lc)
        # [b, num_heads, lq, lc]
        weights = torch.softmax(torch.sum(product, dim=-2), dim=-1)
        # [b, num_heads, lc]
        w_sum = weights.unsqueeze(2) @ m_proj
        # [b, num_heads, 1, lc]
        output = w_sum.squeeze(-2).transpose(-1, -2) @ self.W_0
        # [b, lc, 1]

        return torch.softmax(output.squeeze(-1), dim=-1)
        # [b, lc]


class SelfAttnContxRetrv(torch_nn.Module):
    def __init__(self, lq, lc, num_heads, d_bert, device):
        super().__init__()

        self.lq = lq
        self.lc = lc

        d_k = d_bert // num_heads
        self.W_q = Parameter(torch.rand(num_heads, d_bert, d_k, device=device))
        self.W_c = Parameter(torch.rand(num_heads, d_bert, d_k, device=device))
        self.W_m = Parameter(torch.rand(num_heads, lq, lq, device=device))
        self.W_0 = Parameter(torch.rand(num_heads, 1, device=device))
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.lq, self.lq),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(lc),
        )

    def forward(self, q, c, m):
        # q: [b, lq, d_bert]
        # c: [b, nc, lc, d_bert]
        # m: [lq, lq]

        b, nc, _, _ = c.size()

        q_proj = q.unsqueeze(1) @ self.W_q
        # [b, num_heads, lq, d_k]
        c_proj = c.unsqueeze(2) @ self.W_c
        # [b, nc, num_heads, lc, d_k]
        m_proj = m.view(1, 1, self.lq, self.lq) @ self.W_m
        # [b, num_heads, lq, lq]

        product = c_proj @ q_proj.unsqueeze(1).transpose(-1, -2)
        # [b, nc, num_heads, lc, lq]
        product = self.ff1(product.view(-1, self.lc, self.lq)).view(b, nc, -1, self.lc, self.lq)
        # [b, nc, num_heads, lc, lq]
        weights = torch.softmax(torch.sum(product, dim=-2), dim=-1)
        # [b, nc, num_heads, lq]
        w_sum = weights.unsqueeze(3) @ m_proj.unsqueeze(1)
        # [b, nc, num_heads, 1, lq]
        output = w_sum.squeeze(-2).transpose(-1, -2) @ self.W_0
        # [b, nc, lq, 1]

        return torch.softmax(output.squeeze(-1).sum(1), dim=-1)
        # [b, lq]


class PersistentMemoryCell(torch_nn.Module):
    def __init__(self, lq, lc, num_heads, d_bert, device):
        super().__init__()

        self.query_retrv = SelfAttnQueryRetrv(
            lq=lq, lc=lc, num_heads=num_heads, d_bert=d_bert, device=device
        )
        self.mem_c_retrv = Parameter(torch.rand(lc, lc))
        # self.contx_retrv = SelfAttnContxRetrv(
        #     lq=lq, lc=lc, num_heads=num_heads, d_bert=d_bert, device=device
        # )
        # self.mem_q_retrv = Parameter(torch.rand(lq, lq))

    def forward(self, q, c):
        # q: [b, lq_, d_bert]
        # c: [b, lc_, d_bert]

        ## Retrieve filters from long-term memory and apply them
        c_filter = self.query_retrv(q=q, c=c, m=self.mem_c_retrv)
        # [b, lc_]
        # NOTE: Temporarily not use q_filter
        # q_filter = self.contx_retrv(q=q, c=c, m=self.mem_q_retrv)
        # [b, lq_]

        return c_filter
