import torch
import torch.nn as torch_nn
from transformers import BertModel, BertTokenizer, logging
from utils.model_utils import TransEncoder, ipot

logging.set_verbosity_error()


from models.layers import (
    AnsEncoder,
    DualAttention,
    Embedding,
    ModelingEncoder,
    PersistentMemoryCell,
    ShortTermMemoryCell,
)


class Backbone(torch_nn.Module):
    def __init__(
        self,
        lq,
        lc,
        d_bert,
        d_hid,
        d_vocab,
        dropout,
        num_heads,
        num_heads_persistent,
        num_layers_p,
        num_layers_q,
        num_layers_a,
        path_pretrained,
        criterion,
        device,
    ):
        super().__init__()

        self.lq = lq
        self.lc = lc
        self.la = -1
        tokenizer = BertTokenizer.from_pretrained(path_pretrained)
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id

        self.lin = torch_nn.Linear(d_bert, d_hid)
        self.embedding = Embedding(lq, lc, d_hid, num_heads_persistent, path_pretrained, device)
        self.dual_attn = DualAttention(d_hid)
        self.model_enc = ModelingEncoder(num_layers_p, num_layers_q, num_heads, d_hid, dropout)
        self.ans_enc = AnsEncoder(
            d_hid, num_layers=num_layers_a, num_heads=num_heads, dropout=dropout
        )
        self.shortterm_mem = ShortTermMemoryCell(d_hid, dropout)
        self.decoder = torch_nn.Linear(d_hid, d_vocab)

        self.criterion = criterion

        ## Init
        # self.decoder.weight = self.embedding.encoder.embeddings.word_embeddings.weight
        self.decoder.bias.data.zero_()

    # np: no pad token
    # lq_np : seq len of question (no pad)
    # lc_np : seq len of context (no pad)
    # la_np : seq len of answer (no pad)
    # l_qc = 1 + lq + 1 + lc + 1
    # l_qca = 1 + lq + 1 + lc + 1 + la + 1
    def get_loss(
        self,
        output_mle,
        trgs,
        is_loss_ot=False,
        gamma=0.08,
        a1_ids=None,
        a1_masks=None,
        a2_ids=None,
        a2_masks=None,
        eta=0.1,
        is_loss_bert=False,
    ):
        loss = 0
        for output, trg in zip(output_mle, trgs):
            loss_mle = self.criterion(output, trg)

            loss += loss_mle

            if is_loss_ot:
                trg = self.encoder(trg)[0]
                pred = (
                    torch.softmax(output.transpose(-1, -2), dim=-1)
                    @ self.encoder.embeddings.word_embeddings.weight
                )
                loss_ot = ipot(pred, trg, max_iter=400)
            else:
                loss_ot = 0
            loss += gamma * loss_ot

            if is_loss_bert:
                output = torch.softmax(output.transpose(-1, -2), dim=-1)
                loss_bert = eta * self.bertloss(
                    pred=output[:, 1:-1],
                    a1_ids=a1_ids,
                    a1_masks=a1_masks,
                    a2_ids=a2_ids,
                    a2_masks=a2_masks,
                )
                loss += loss_bert

        return loss / len(trgs)

    def forward(self, q_ids, c_ids, a_ids):
        # q: [b, lq]
        # c: [b, nc, lc]
        # a: [b, la]

        q_mem, c_mem, a_mem = None, None, None
        for i in range(c_ids.size(1)):
            ## Encode q, c and a using Bert
            q, ci, a, trgs = self.embedding(q_ids, c_ids[:, i], a_ids, a_ids.size(-1))
            # [b, l_, d_bert]

            q, ci, a = self.lin(q), self.lin(ci), self.lin(a)

            ## Use Dual Attention
            G_pq, G_qp = self.dual_attn(q, ci)
            # G_pq: [b, lq, 5d]
            # G_qp: [b, lc, 5d]

            ## Use Modeling Encoder
            Mq, Mp = self.model_enc(G_pq, G_qp)
            # Mq: [b, lq, d]
            # Mp: [b, lc, d]

            ## Encode answer with q and c
            a = self.ans_enc(a, Mq, Mp)
            # [b, la, d]

            # Apply Short-term Memory
            q_mem, c_mem, a_mem = self.shortterm_mem(q, ci, a, q_mem, c_mem, a_mem)

        output_mle = self.decoder(a_mem)
        # [b, la + 2, d_vocab]

        return output_mle.transpose(1, 2), trgs

    def do_train(self, q, c, a1, a2=None, a1_mask=None, a2_mask=None, use_2_ans=False):
        output_mle, trgs = [], []
        ans = [a1, a2] if use_2_ans else [a1]
        for a in ans:
            output_mle_, trgs_ = self(q, c, a)

            output_mle.append(output_mle_)
            trgs.append(trgs_)

        # NOTE: Temporarily not using OT loss and BERT loss
        loss = self.get_loss(output_mle=output_mle, trgs=trgs)

        return loss, output_mle

    def do_predict(self, q, c, la):
        bz = q.size(0)

        ans = []
        for i in range(bz):
            q_, c_ = q[i : i + 1], c[i : i + 1]
            a = torch.tensor([[]], device=q.device)

            for _ in range(la):
                output, _ = self(q_, c_, a)
                topi = torch.log_softmax(output[:, :, a.size(-1)], dim=-1).argmax(dim=-1)
                if topi == self.sep_id:
                    break
                a = torch.cat((a, topi.unsqueeze(-1)), dim=-1)

            a_ = torch.full((1, la), self.pad_id, dtype=a.dtype, device=a.device)
            a_[0, : a.size(-1)] = a[0]
            ans.append(a_)

        return torch.cat(ans, dim=0)
