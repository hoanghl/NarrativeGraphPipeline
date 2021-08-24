import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_f
import transformers
from transformers import BertModel

transformers.logging.set_verbosity_error()


class FineGrain(torch_nn.Module):
    """Embed and generate question-aware c"""

    def __init__(
        self,
        l_a,
        n_gru_layers,
        d_bert,
        d_hid,
        path_pretrained,
    ):
        super().__init__()

        self.d_bert = d_bert

        self.bert_emb = BertModel.from_pretrained(path_pretrained)
        self.biGRU_CoAttn = torch_nn.GRU(
            d_bert,
            d_bert // 2,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.lin1 = torch_nn.Sequential(
            torch_nn.Linear(d_bert, d_bert),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(l_a - 1),
            torch_nn.Linear(d_bert, d_hid // 2),
        )

    def remove_special_toks(self, a, a_masks):
        bz = a.size(0)

        a_ = torch.zeros_like(a)

        for i in range(bz):
            l = a_masks[i].sum() - 2
            a_[i, :l] = a[i, 1 : 1 + l]

        return a_

    def encode_ques_para(self, q_ids, c_ids, q_masks, c_masks):
        # q_ids: [b, l_q]
        # c_ids: [b, n_c, l_c]
        # q_masks: [b, l_q]
        # c_masks: [b, n_c, l_c]

        n_c = c_ids.size(1)

        q = self.bert_emb(input_ids=q_ids, attention_mask=q_masks)[0]
        # [b, l_q, d_bert]
        q = self.remove_special_toks(q, q_masks)
        # [b, l_q, d_bert]

        #########################
        # Operate CoAttention question
        # with each paragraph
        #########################
        paragraphs = []

        for ith in range(n_c):
            contx = c_ids[:, ith, :]
            contx_mask = c_masks[:, ith, :]

            ###################
            # Embed c
            ###################
            L_s = self.bert_emb(input_ids=contx, attention_mask=contx_mask)[0]
            # [b, l_c, d_bert]
            L_s = self.remove_special_toks(L_s, contx_mask)
            # [b, l_c, d_bert]

            ###################
            # Operate CoAttention between
            # query and c
            ###################

            # Affinity matrix
            A = torch.bmm(L_s, q.transpose(1, 2))
            # A: [b, l_c, l_q]

            # S_s  = torch.matmul(torch_f.softmax(A, dim=1), E_q)
            S_q = torch.bmm(torch_f.softmax(A.transpose(1, 2), dim=1), L_s)
            # S_q: [b, l_q, d_bert]

            X = torch.bmm(torch_f.softmax(A, dim=1), S_q)
            C_s = self.biGRU_CoAttn(X)[0]

            C_s = torch.unsqueeze(C_s, 1)
            # C_s: [b, 1, l_c, d_bert]

            paragraphs.append(C_s)

        c = torch.cat(paragraphs, dim=1)
        # [b, n_c, l_c, d_bert]

        return q, c

    def encode_ans(self, a_ids, a_masks, ot_loss=False):
        # a_ids: [b, l_a]
        # a_masks: [b, l_a]

        output = self.bert_emb(input_ids=a_ids, attention_mask=a_masks)[0]
        # [b, l_a, d_bert]
        if ot_loss:
            output = self.lin1(output)
        # [b, l_a, d_hid]

        return output

    def get_output_ot(self, output):
        # output: [b, l_a, d_vocab]

        output_ot = output @ self.bert_emb.embeddings.word_embeddings.weight
        # [b, l_a, d_bert]
        output_ot = self.lin1(output_ot)
        # [b, l_a, d_hid]

        return output_ot
