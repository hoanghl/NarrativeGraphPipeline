from transformers import BertModel
import transformers
import torch.nn as torch_nn
import torch

transformers.logging.set_verbosity_error()


class BertBasedEmbedding(torch_nn.Module):
    """Embed and generate question-aware c"""

    def __init__(self, d_bert, d_hid, path_pretrained):
        super().__init__()

        self.d_bert = d_bert

        ## Modules for embedding
        self.bert_emb = BertModel.from_pretrained(path_pretrained)
        self.lin1 = torch_nn.Linear(d_bert, d_hid)

    def forward(self):
        return

    def encode_ques_para(self, q_ids, c_ids, q_masks, c_masks):
        # q: [b, l_q]
        # c_ids: [b, n_c, l_c]
        # q_masks: [b, l_q]
        # c_masks: [b, n_c, l_c]

        b, _, l_c = c_ids.shape

        #########################
        # Contextual embedding for question with BERT
        #########################
        q = self.bert_emb(input_ids=q_ids, attention_mask=q_masks)[0]
        # [b, l_q, d_bert]

        #########################
        # Contextual embedding for c with BERT
        #########################
        # Convert to another shape to fit with
        # input shape of self.embedding
        c = c_ids.view((-1, l_c))
        c_masks = c_masks.view((-1, l_c))
        # c     : [b*n_c, l_c, d_bert]
        # c_masks: [b*n_c, l_c]

        c = self.bert_emb(input_ids=c, attention_mask=c_masks)[0]
        # [b*n_c, l_c, d_bert]
        c = c.view((b, -1, l_c, self.d_bert))
        # [b, n_c, l_c, d_bert]

        q, c = self.lin1(q), self.lin1(c)

        return q, c

    def encode_ans(self, input_ids=None, input_embds=None, input_masks=None):
        # input_ids : [b, len_]
        # input_embds : [b, len_, d_bert]
        # input_mask : [b, len_]

        assert torch.is_tensor(input_ids) or torch.is_tensor(
            input_embds
        ), "One of two must not be None"

        encoded = (
            self.bert_emb(
                input_ids=input_ids,
                attention_mask=input_masks,
            )[0]
            if torch.is_tensor(input_ids)
            else self.bert_emb(
                inputs_embeds=input_embds,
                attention_mask=input_masks,
            )[0]
        )
        # [b, len_, d_bert]

        encoded = self.lin1(encoded)
        # [b, len_, d_hid]

        return encoded

    def get_output_ot(self, output):
        # output: [b, l_a - 1, d_vocab]

        output = output @ self.bert_emb.embeddings.word_embeddings.weight
        # [b, l_a - 1, d_bert]

        output = self.lin1(output)
        # [b, l_a - 1, d_hid]

        return output
