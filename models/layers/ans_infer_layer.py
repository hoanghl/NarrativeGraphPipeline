import numpy as np
import torch
import torch.nn as torch_nn
from transformers import BertConfig, BertModel


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        l_a,
        d_bert,
        d_vocab,
        tokenizer,
        embd_layer,
    ):
        super().__init__()

        self.cls_tok_id = tokenizer.cls_token_id
        self.sep_tok_id = tokenizer.sep_token_id

        self.d_vocab = d_vocab
        self.l_a = l_a

        self.embd_layer = embd_layer

        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.add_cross_attention = True
        self.decoder = BertModel(config=bert_conf)

        self.ff = torch_nn.Linear(d_bert, d_vocab)

    def forward(self, Y, a_ids, a_masks=None):
        # Y: [b, n_c*l_c, d_bert]
        # a_ids : [b, seq_len]
        # a_masks: [b, seq_len]

        input_embds = self.embd_layer.encode_ans(a_ids=a_ids, a_masks=a_masks)

        output = self.decoder(
            inputs_embeds=input_embds, attention_mask=a_masks, encoder_hidden_states=Y
        )[0]
        # [b, l_a, 768]

        pred = self.ff(output)
        # [b, l_a, d_vocab]

        return pred

    def do_train(
        self,
        Y,
        a_ids,
        a_masks,
        cur_step: int,
        max_step: int,
    ):
        # Y: [b, n_c*l_c, d_bert]
        # a_ids: [b, l_a]
        # a_masks: [b, l_a]

        input_ids = torch.full((Y.size()[0], 1), self.cls_tok_id, device=Y.device)
        # [b, 1]

        for ith in range(1, self.l_a + 1):
            output = self(Y=Y, a_ids=input_ids, a_masks=a_masks[:, :ith])
            # [b, ith, d_vocab]

            ## Apply Scheduling teacher
            if ith == self.l_a:
                break

            _, topi = torch.topk(torch.softmax(output[:, -1, :], dim=-1), k=1)
            chosen = self.choose_scheduled_sampling(
                output=topi,
                a_ids=a_ids,
                ith=ith,
                cur_step=cur_step,
                max_step=max_step,
            )
            # [b]
            input_ids = torch.cat((input_ids, chosen.detach()), dim=1)

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(torch.softmax(output, dim=-1))[:, :-1]
        # [b, l_a - 1, d_hid]

        output_mle = output[:, :-1].transpose(1, 2)
        # [b, d_vocab, l_a - 1]

        return output_mle, output_ot

    def choose_scheduled_sampling(
        self,
        output,
        a_ids,
        ith: int,
        cur_step: int,
        max_step,
    ):

        # output: [b, 1]
        # a_ids   : [b, l_a]

        t = (
            np.random.binomial(1, cur_step / max_step)
            if max_step != 0
            else np.random.binomial(1, 1)
        )

        return a_ids[:, ith].unsqueeze(1) if t == 0 else output

    def do_predict(self, Y):
        # Y: [b, n_c*l_c, d_bert]

        b = Y.size(0)

        ## Init input_embs with cls embedding
        cls_ids = torch.full(
            (b,),
            fill_value=self.cls_tok_id,
            device=Y.device,
            requires_grad=False,
        )
        input_ids = [cls_ids.unsqueeze(1)]

        for ith in range(1, self.l_a + 1):
            output = self(Y=Y, a_ids=torch.cat(input_ids, dim=1))
            # [b, ith, d_vocab]

            if ith == self.l_a:
                break

            _, topi = torch.topk(torch.softmax(output[:, -1, :], dim=-1), k=1)

            input_ids.append(topi.detach())

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(torch.softmax(output, dim=-1))[:, :-1]
        # [b, l_a - 1, d_hid]

        output_mle = output[:, :-1].transpose(1, 2)
        # [b, d_vocab, l_a - 1]

        return output_mle, output_ot
