import random

from transformers import BertConfig, BertModel
import torch.nn as torch_nn
import torch
import numpy as np

from utils.model_utils import GeneratorOwn


class BertDecoder(torch_nn.Module):
    def __init__(
        self,
        l_a,
        d_bert,
        d_vocab,
        cls_tok_id,
        sep_tok_id,
        beam_size,
        n_gram_beam,
        embd_layer,
    ):
        super().__init__()

        self.beam_size = beam_size
        self.cls_tok_id = cls_tok_id
        self.sep_tok_id = sep_tok_id
        self.n_gram_beam = n_gram_beam
        self.l_a = l_a
        self.t = -1

        self.embd_layer = embd_layer

        bert_conf = BertConfig()
        bert_conf.is_decoder = True
        bert_conf.add_cross_attention = True
        bert_conf.num_attention_heads = 6
        bert_conf.num_hidden_layers = 6
        self.decoder = BertModel(config=bert_conf)

        self.ff = torch_nn.Sequential(
            torch_nn.Linear(d_bert, d_bert),
            torch_nn.GELU(),
            torch_nn.Linear(d_bert, d_vocab),
        )

    def forward(self, Y: torch.Tensor, ans_ids: torch.Tensor, ans_mask: torch.Tensor):
        # Y       : [b, l_a, d_bert]
        # ans_ids : [b, seq_len]
        # ans_mask: [b, seq_len]

        input_embds = self.embd_layer.encode_ans(ans_ids=ans_ids, ans_mask=ans_mask)

        output = self.decoder(
            inputs_embeds=input_embds, attention_mask=ans_mask, encoder_hidden_states=Y
        )[0]
        # [b, l_a, 768]

        pred = self.ff(output)
        # [b, l_a, d_vocab]

        return pred

    def do_train(
        self,
        Y: torch.Tensor,
        ans_ids: torch.Tensor,
        ans_mask: torch.Tensor,
        cur_step: int,
        max_step: int,
    ):
        # Y         : [b, l_c, d_hid]
        # ans_ids   : [b, l_a]
        # ans_mask  : [b, l_a]

        input_ids = torch.full((Y.size()[0], 1), self.cls_tok_id, device=Y.device)
        # [b, 1]

        for ith in range(1, self.l_a + 1):
            output = self(Y=Y, ans_ids=input_ids, ans_mask=ans_mask[:, :ith])
            # [b, ith, d_vocab]

            ## Apply Scheduling teacher
            if ith == self.l_a:
                break

            _, topi = torch.topk(torch.softmax(output[:, -1, :], dim=-1), k=1)
            chosen = self.choose_scheduled_sampling(
                output=topi,
                ans_ids=ans_ids,
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
        output: torch.Tensor,
        ans_ids: torch.Tensor,
        ith: int,
        cur_step: int,
        max_step,
    ):

        # output: [b, 1]
        # ans_ids   : [b, l_a]

        self.t = (
            np.random.binomial(1, cur_step / max_step)
            if max_step != 0
            else np.random.binomial(1, 1)
        )

        return ans_ids[:, ith].unsqueeze(1) if self.t == 0 else output

    def do_predict(self, Y):
        # NOTE: This belongs to BeamSearchHugging and therefore is commented
        # Y_ = Y.repeat_interleave(self.beam_size, dim=0)
        # # [b_, l_a, d_bert]

        # generator = BeamSearchHuggingface(
        #     batch_size=b,
        #     max_length=self.l_a,
        #     num_beams=self.beam_size,
        #     temperature=self.temperature,
        #     no_repeat_ngram_size=self.n_gram_beam,
        #     model=self.generate,
        #     pad_token_id=self.bert_tokenizer.pad_token_id,
        #     bos_token_id=self.bert_tokenizer.cls_token_id,
        #     eos_token_id=self.bert_tokenizer.sep_token_id,
        # )

        # outputs = generator.beam_sample(None, Y_)

        b = Y.size(0)

        output_ids = []

        beam_search = GeneratorOwn(
            beam_size=self.beam_size,
            init_tok=self.cls_tok_id,
            stop_tok=self.sep_tok_id,
            max_len=self.l_a,
            model=self.generate_own,
            no_repeat_ngram_size=self.n_gram_beam,
        )

        for b_ in range(b):
            indices = beam_search.search(encoder_outputs=Y[b_, :, :])
            output_ids.append(indices)
        output_ids = torch.LongTensor(output_ids, device=Y.device)[:, 1:]
        # [b, l_a]

        output_mask = torch.zeros(output_ids.size(), device=Y.device)
        output = self(Y=Y, ans_ids=output_ids, ans_mask=output_mask)

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(torch.softmax(output, dim=-1))[:, :-1]
        # [b, l_a - 1, d_hid]

        output_mle = output[:, :-1].transpose(1, 2)
        # [b, d_vocab, l_a - 1]

        return output_mle, output_ot

    def generate_own(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [l_]
        # encoder_outputs  : [l_a, d_bert]

        decoder_input_ids = torch.LongTensor(
            decoder_input_ids, device=encoder_outputs.device
        ).unsqueeze(0)
        decoder_input_mask = torch.ones(
            decoder_input_ids.size(), device=encoder_outputs.device
        )
        encoder_outputs = encoder_outputs.unsqueeze(0)

        output = self(encoder_outputs, decoder_input_ids, decoder_input_mask)
        # [1, l_, d_vocab]

        output = output.squeeze(0)
        # [l_, d_vocab]

        return output

    # NOTE: This belongs to BeamSearchHugging and therefore is commented
    # def generate(self, decoder_input_ids, encoder_outputs):
    #     # decoder_input_ids: [b_, seq_len<=200]
    #     # encoder_outputs  : [b_, len_, d_bert]

    #     b_, seq_len = decoder_input_ids.shape

    #     decoder_input_mask = torch.ones((b_, seq_len))
    #     decoder_input_embd = self.embd_layer.encode_ans(
    #         decoder_input_ids, decoder_input_mask
    #     )
    #     # [b_, seq=*, d_bert]

    #     output = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
    #     # [b_, seq=*, d_vocab]

    #     return Seq2SeqLMOutput(logits=output)
