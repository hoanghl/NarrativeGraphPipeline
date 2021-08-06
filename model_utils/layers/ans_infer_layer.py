from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch
import numpy as np


from utils.model_utils import GeneratorHugging


class Decoder(torch_nn.Module):
    def __init__(
        self, batch_size, l_a, d_vocab, d_hid, tokenizer, embd_layer, beam_size, device
    ):
        super().__init__()

        self.l_a = l_a
        self.tokenizer = tokenizer
        self.d_vocab = d_vocab

        self.t = 0
        self.beam_size = beam_size

        self.embd_layer = embd_layer
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
        self.generator = GeneratorHugging(
            batch_size=batch_size,
            max_length=l_a,
            min_length=1,
            num_beams=beam_size,
            temperature=0.8,
            no_repeat_ngram_size=5,
            model=self.generate,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            device=device,
        )

    def forward(self, Y, input_ids, input_masks):
        # Y: [b, 1, d_hid]
        # input_ids: [b, l_,]
        # input_masks: [b, l_]

        encoded = self.embd_layer.encode_ans(
            input_ids=input_ids, input_masks=input_masks
        )
        # [b_, l_, d_hid]

        output = self.trans_decoder(
            inputs_embeds=encoded,
            attention_mask=input_masks,
            encoder_hidden_states=Y,
        )[0]
        # [b_, l_, d_hid]

        output = self.lin1(output)
        # [b_, l_, d_vocab]

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

        for ith in range(1, self.l_a + 1):
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

        return output

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
    def generate(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [batch_beam, l_]
        # encoder_outputs  : [batch_beam, l_c, d_]

        decoder_input_mask = torch.ones(
            decoder_input_ids.shape, device=encoder_outputs.device
        )

        output = self(encoder_outputs, decoder_input_ids, decoder_input_mask)
        # [b_, len_, d_vocab]

        return output

    def do_predict(self, Y):
        Y_ = Y.repeat_interleave(self.beam_size, dim=0)
        # [b_, l_a, d_hid]

        output = self.generator.beam_sample(None, Y_)
        # [b, l_a]
        output = ids2dist(output, self.d_vocab)
        # [b, l_a, d_vocab]

        return output


def ids2dist(outputs, d_vocab):
    indices = outputs.unsqueeze(-1)
    a = torch.full((*outputs.size(), d_vocab), 1e-6, device=outputs.device)
    a.scatter_(
        dim=-1,
        index=indices,
        src=torch.full(indices.size(), 0.99, device=outputs.device),
    )
    return a
