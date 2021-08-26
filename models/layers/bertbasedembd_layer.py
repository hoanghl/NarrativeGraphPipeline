import torch
import torch.nn as torch_nn
import transformers
from transformers import BertModel

transformers.logging.set_verbosity_error()


class BertBasedEmbedding(torch_nn.Module):
    """Embed and generate question-aware c"""

    def __init__(self, d_bert, d_hid, path_pretrained):
        super().__init__()

        self.d_bert = d_bert

        ## Modules for embedding
        self.bert_emb = BertModel.from_pretrained(path_pretrained)
        self.lin1 = torch_nn.Linear(d_bert, d_hid, bias=False)

    def remove_special_toks(self, a, a_masks):
        bz = a.size(0)

        a_ = torch.zeros_like(a)

        for i in range(bz):
            l = a_masks[i].sum() - 2
            l_pad = a_masks.size(-1) - a_masks[i].sum()
            a_[i, : l + l_pad] = torch.cat((a[i, 1 : 1 + l], a[i, l + 2 :]), dim=0)

        return a_

    def encode_ques_para(self, q_ids, c_ids, q_masks, c_masks):
        # q: [b, lq]
        # c_ids: [b, n_c, lc]
        # q_masks: [b, lq]
        # c_masks: [b, n_c, lc]

        b, _, lc = c_ids.shape

        #########################
        # Contextual embedding for question with BERT
        #########################
        q = self.bert_emb(input_ids=q_ids, attention_mask=q_masks)[0]
        # [b, lq, d_bert]
        q = self.remove_special_toks(q, q_masks)

        #########################
        # Contextual embedding for c with BERT
        #########################
        # Convert to another shape to fit with
        # input shape of self.embedding
        c = c_ids.view((-1, lc))
        # [b*n_c, lc, d_bert]
        c_masks = c_masks.view((-1, lc))
        # [b*n_c, lc]
        c = self.bert_emb(input_ids=c, attention_mask=c_masks)[0]
        c = self.remove_special_toks(c, c_masks)
        # [b*n_c, lc, d_bert]
        c = c.view((b, -1, lc, self.d_bert))
        # [b, n_c, lc, d_bert]

        q, c = self.lin1(q), self.lin1(c)

        return q, c

    def encode_tok(self, tok):
        # tok: [d_vocab]
        return tok @ self.bert_emb.embeddings.word_embeddings.weight

    def encode_ans(self, a_ids, a_masks=None, ot_loss=False):
        # a_ids: [b, la]
        # a_masks: [b, la]

        output = self.bert_emb(input_ids=a_ids, attention_mask=a_masks)[0]
        # [b, la, d_bert]

        output = self.lin1(output)
        # [b, la, d_hid]

        return output

    def get_output_ot(self, output):
        # output: [b, la, d_vocab]

        output_ot = output @ self.bert_emb.embeddings.word_embeddings.weight
        # [b, la, d_bert]
        output_ot = self.lin1(output_ot)
        # [b, la, d_hid]

        return output_ot
