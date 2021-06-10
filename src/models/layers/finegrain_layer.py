from transformers import BertModel
import transformers
import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

transformers.logging.set_verbosity_error()


class FineGrain(torch_nn.Module):
    """Embed and generate question-aware context"""

    def __init__(
        self,
        seq_len_para: int = 182,
        n_gru_layers: int = 5,
        d_bert: int = 768,
        path_bert: str = None,
    ):
        super().__init__()

        self.d_bert = d_bert

        self.bert_emb = BertModel.from_pretrained(path_bert)
        self.biGRU_CoAttn = torch_nn.GRU(
            d_bert,
            d_bert // 2,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.biGRU_attn = torch_nn.GRU(
            d_bert,
            d_bert // 2,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.biGRU_mask = torch_nn.GRU(
            d_bert,
            d_bert // 2,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lin_attn = torch_nn.Linear(d_bert * 2, seq_len_para)

    def forward(self):
        return

    def encode_ques_para(self, ques, paras, ques_mask, paras_mask):
        # ques          : [b, seq_len_ques]
        # paras         : [b, n_paras, seq_len_para]
        # ques_mask     : [b, seq_len_ques]
        # paras_mask    : [b, n_paras, seq_len_para]

        b, n_paras, seq_len_para = paras.shape

        ques = self.bert_emb(ques, ques_mask)[0]
        # ques  : [b, seq_len_ques, d_bert]

        #########################
        # Operate CoAttention question
        # with each paragraph
        #########################
        paragraphs = []

        for ith in range(n_paras):
            para = paras[:, ith, :, :]
            para_mask = paras_mask[:, ith, :]

            ###################
            # Embed context
            ###################
            L_s = self.bert_emb(para, para_mask)[0]
            # L_s: [b, seq_len_para, d_bert]

            ###################
            # Operate CoAttention between
            # query and context
            ###################

            # Affinity matrix
            A = torch.bmm(L_s, ques.transpose(1, 2))
            # A: [b, seq_len_para, seq_len_ques]

            # S_s  = torch.matmul(torch_f.softmax(A, dim=1), E_q)
            S_q = torch.bmm(torch_f.softmax(A.transpose(1, 2), dim=1), L_s)
            # S_q: [b, seq_len_ques, d_bert]

            X = torch.bmm(torch_f.softmax(A, dim=1), S_q)
            C_s = self.biGRU_CoAttn(X)[0]

            C_s = torch.unsqueeze(C_s, 1)
            # C_s: [b, 1, seq_len_para, d_bert]

            paragraphs.append(C_s)

        paras = torch.cat((paragraphs), dim=1)
        # paras : [b, n_paras, seq_len_para, d_bert]

        #########################
        # Reduce 'paras' by applying attentive method
        # based on 'ques'
        #########################
        ques_ = torch.mean(ques, dim=1)
        # [b, d_bert]

        paras_ = paras.reshape(-1, seq_len_para, self.d_bert)
        # paras_len = torch.sum(paras_mask, dim=2).reshape(-1).to("cpu")
        # paras_    : [b*n_paras, seq_len_para, d_bert]
        # paras_mask: [b*n_paras]

        # for i in range(paras_len.shape[0]):
        #     if paras_len[i] == 0:
        #         paras_len[i] = 1

        # tmp     = torch_nn.utils.rnn.pack_padded_sequence(paras_, paras_len, batch_first=True,
        #                                                   enforce_sorted=False)
        # tmp     = self.biGRU_mask(tmp)[0]
        # paras_  = torch_nn.utils.rnn.pad_packed_sequence(tmp, batch_first=True)[0]
        paras_ = self.biGRU_mask(paras_)[0]
        # [b*n_paras, seq_len_para, d_bert]

        paras_first = paras_[:, 0, :].reshape(b, n_paras, -1)
        # [b, n_paras, d_bert]

        q_ = ques_.unsqueeze(1).repeat(1, n_paras, 1)
        # [b, n_paras, d_bert]
        selector = torch.cat((q_, paras_first), dim=2)
        # [b, n_paras, d_bert*2]
        selector = self.lin_attn(selector)
        # [b, n_paras, seq_len_para]

        selector = selector.unsqueeze(3).repeat(1, 1, 1, self.d_bert)
        # [b, n_paras, seq_len_para, d_bert]

        paras = torch.sum(paras * selector, dim=2)
        # [b, n_paras, d_bert]

        return ques, paras

    def encode_ans(self, ans, ans_mask):
        # ans           : [b, seq_len_ans]
        # ans_mask      : [b, seq_len_ans]

        return self.bert_emb(ans, ans_mask)[0]
        # [b, seq_len_ans, d_bert]
