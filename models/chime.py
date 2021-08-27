import torch
import torch.nn as torch_nn
from transformers import BertModel, BertTokenizer, logging
from utils.model_utils import TransEncoder, ipot

logging.set_verbosity_error()


class CHIME(torch_nn.Module):
    def __init__(self, lq, lc, d_bert, d_vocab, path_pretrained, criterion):
        super().__init__()

        self.lq = lq
        self.lc = lc
        self.la = -1
        tokenizer = BertTokenizer.from_pretrained(path_pretrained)
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

        self.encoder = BertModel.from_pretrained(path_pretrained)
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
    #

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

        part1, part2 = [], []
        for output, lq_, lc_, la_ in zip(outputs, lq_np, lc_np, la_np):
            l_pad_q = self.lq - lq_
            l_pad_c = self.lc - lc_

            l1 = 1 + lq_ + 1
            l2 = 1 + lq_ + 1 + lc_ + 1 + la_ + 1
            p1 = torch.cat(
                [output[1 : 1 + lq_], output[l1 : l1 + lc_], output[l2 : l2 + l_pad_q + l_pad_c]],
                dim=0,
            )
            l1 = 1 + lq_ + 1 + lc_
            l2 = 1 + lq_ + 1 + lc_ + 1 + la_ + 1 + l_pad_q + l_pad_c
            p2 = torch.cat([output[l1 : l1 + 1 + la_ + 1], output[l2:]], dim=0)

            part1.append(p1.unsqueeze(0))
            part2.append(p2.unsqueeze(0))

        return torch.cat(part1, dim=0), torch.cat(part2, dim=0)

    def forward(self, q, c, a):
        # q: [b, lq]
        # c: [b, nc, lc]
        # a: [b, la]

        self.la = a.size(-1)

        c_mem, a_mem = None, None
        for n in range(c.size(1)):
            ####################
            # From q, c and a, create
            # id, mask and tok_type_id tensors
            ####################
            qca_ids, qca_masks, qca_tok_type_id, trgs, lq_np, lc_np, la_np = self.create_misc(
                q, c[:, n], a
            )

            ####################
            # Encode qca using Bert
            ####################
            part1, part2 = self.encode(qca_ids, qca_masks, qca_tok_type_id, lq_np, lc_np, la_np)

            ####################
            # Apply TransEncoder with memory
            ####################
            if not torch.is_tensor(c_mem):
                c_mem, a_mem = part1, part2
            else:
                ## Apply TransEncoder with part1
                z = self.c_mem_encoder(c_mem, part1, part1)
                c_gate = self.gate_c(torch.cat((z, c_mem), dim=-1))
                c_mem = c_gate * c_mem + (1 - c_gate) * part1

                ## Apply TransEncoder with part2
                z = self.a_mem_encoder(a_mem, c_mem, c_mem)
                a_gate = self.gate_a(torch.cat((z, a_mem), dim=-1))
                a_mem = a_gate * a_mem + (1 - a_gate) * part2

        output_mle = self.decoder(a_mem)
        # [b, la + 2, d_vocab]

        return output_mle.transpose(1, 2), trgs

    def do_train(self, q, c, a1, a2, use_2_ans=False):
        output_mle, trgs = [], []
        ans = [a1, a2] if use_2_ans else [a1]
        for a in ans:
            output_mle_, trgs_ = self(q, c, a)

            output_mle.append(output_mle_)
            trgs.append(trgs_)

        # NOTE: can set is_loss_ot=true
        loss = self.get_loss(output_mle, trgs)

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
