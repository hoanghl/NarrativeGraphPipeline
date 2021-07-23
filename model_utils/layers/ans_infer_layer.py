from typing import Any


from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch
import numpy as np


from model_utils.layers.bertbasedembd_layer import BertBasedEmbedding


class Decoder(torch_nn.Module):
    def __init__(
        self,
        l_a,
        d_vocab,
        d_hid,
        tokenizer,
        embd_layer,
    ):
        super().__init__()

        self.l_a = l_a
        self.d_hid = d_hid
        self.tokenizer = tokenizer
        self.d_vocab = d_vocab

        self.t = 0

        self.embd_layer: BertBasedEmbedding = embd_layer
        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.hidden_size = d_hid
        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 8
        bert_conf.num_hidden_layers = 6
        self.trans_decoder = BertModel(config=bert_conf)

        self.lin1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Tanh(),
            torch_nn.Linear(d_hid, d_vocab),
        )

    def forward(self, Y, input_ids, input_masks):
        # Y: [b, 1, d_hid]
        # input_ids: [b, l_,]
        # input_masks: [b, l_]

        encoded = self.embd_layer.encode_ans(
            input_ids=input_ids, input_masks=input_masks
        )

        output = self.trans_decoder(
            inputs_embeds=encoded,
            attention_mask=input_masks,
            encoder_hidden_states=Y,
        )[0]
        # [b, l_, d_hid]

        output = self.lin1(output)
        # [b, l_, d_vocab]

        return output

    def do_train(self, Y, a_ids, a_masks, cur_step, max_step):
        # Y: [b, 1, d_hid]
        # a_ids: [b, l_a]
        # a_masks: [b, l_a]

        b = Y.size(0)

        ## Init input_embs with cls embedding
        cls_ids = torch.full(
            (b,),
            fill_value=self.tokenizer.cls_token_id,
            device=Y.device,
            requires_grad=False,
        )
        input_ids = [cls_ids.unsqueeze(1)]

        for ith in range(1, self.l_a):
            output = self(
                Y=Y,
                input_ids=torch.cat(input_ids, dim=1),
                input_masks=a_masks[:, :ith],
            )
            # [b, ith, d_vocab]

            if ith == self.l_a:
                break

            _, topi = torch.topk(torch.softmax(output[:, -1, :], dim=-1), k=1)
            new_word = self.choose_scheduled_sampling(
                output=topi,
                a_ids=a_ids[:, ith].unsqueeze(-1),
                cur_step=cur_step,
                max_step=max_step,
            )
            # new_word: [b, 1]

            input_ids.append(new_word)

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(output)
        # [b, l_a - 1, d_hid]

        output_mle = output.transpose(1, 2)
        # [b, d_vocab, l_a - 1]

        return output_mle, output_ot

    ##############################################
    # Methods for scheduled sampling
    ##############################################

    def choose_scheduled_sampling(self, output, a_ids, cur_step, max_step):
        # output: [b, 1]
        # a_ids: [b, 1]

        # Apply Scheduled Sampling
        self.t = np.random.binomial(1, cur_step / max_step)

        return a_ids if self.t == 0 else output

    ##############################################
    # Methods for validation/prediction
    ##############################################
    def do_predict(
        self,
        Y: torch.Tensor,
        a_masks: torch.Tensor,
    ):

        b = Y.size(0)

        ## Init input_embs with cls embedding
        cls_ids = torch.full(
            (b,),
            fill_value=self.tokenizer.cls_token_id,
            device=Y.device,
            requires_grad=False,
        )
        input_ids = [cls_ids.unsqueeze(1)]

        for ith in range(1, self.l_a):
            output = self(
                Y=Y,
                input_ids=torch.cat(input_ids, dim=1),
                input_masks=a_masks[:, :ith],
            )
            # [b, ith, d_vocab]

            if ith == self.l_a:
                break

            _, topi = torch.topk(torch.softmax(output[:, -1, :], dim=-1), k=1)
            new_word = topi

            input_ids.append(new_word)

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(output)
        # [b, l_a - 1, d_hid]

        output_mle = output.transpose(1, 2)
        # [b, d_vocab, l_a - 1]

        return output_mle, output_ot
