import numpy as np
import torch
import torch.nn as torch_nn
from transformers import BertConfig, BertModel
from utils.model_utils import ipot


def ids2dist(inputs, d_vocab):
    indices = inputs.unsqueeze(-1)
    a = torch.full((*inputs.size(), d_vocab), 1e-6, dtype=torch.float, device=inputs.device)
    a.scatter_(
        dim=-1,
        index=indices,
        src=torch.full(indices.size(), 0.99, dtype=torch.float, device=inputs.device),
    )
    return a


def is_schedsampl(cur_step, max_step):
    t = np.random.binomial(1, cur_step / max_step) if max_step != 0 else np.random.binomial(1, 1)
    return t == 0


class Decoder(torch_nn.Module):
    def __init__(self, la, d_hid, d_vocab, dropout, tokenizer, embd_layer, criterion):
        super().__init__()

        self.cls_tok_id = tokenizer.cls_token_id
        self.sep_tok_id = tokenizer.sep_token_id
        self.d_hid = d_hid
        self.d_vocab = d_vocab
        self.la = la
        self.criterion = criterion
        self.embd_layer = embd_layer
        bert_conf = BertConfig()
        bert_conf.hidden_size = d_hid
        bert_conf.is_decoder = True
        bert_conf.add_cross_attention = True
        bert_conf.num_hidden_layers = 8
        bert_conf.num_attention_heads = 8
        self.decoder = BertModel(config=bert_conf)
        self.ff = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Tanh(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(d_hid, d_vocab, bias=False),
        )

    def get_loss(self, output_mle, a_ids, a_masks, gamma=0.08):
        # output_mle: [b, la, d_vocab]
        # output_ot: [b, la, d_hid]
        # a_ids: [b, la]
        # a_masks: [b, la]

        trg_ids = torch.zeros_like(a_ids)
        trg_ids[:, :-1] = a_ids[:, 1:]
        trg_masks = torch.zeros_like(a_masks)
        trg_masks[:, :-1] = a_masks[:, 1:]

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(torch.softmax(output_mle, dim=-1))
        # [b, la, d_hid]

        # Calculate MLE loss
        loss_mle = self.criterion(output_mle.transpose(1, 2), trg_ids)

        # Calculate OT loss
        a = self.embd_layer.encode_ans(a_ids=trg_ids, a_masks=trg_masks, ot_loss=True)
        # [b, la-1, d_hid]
        loss_ot = ipot(output_ot, a, max_iter=400)

        total_loss = loss_mle + gamma * loss_ot

        return total_loss

    def forward(self, Y, a_embds, a_masks=None):
        # Y: [b, nc*lc, d_hid]
        # a_ids : [b, seq_len]
        # a_masks: [b, seq_len]

        output = self.decoder(
            inputs_embeds=a_embds, attention_mask=a_masks, encoder_hidden_states=Y
        )[0]
        # [b, la, d_hid]

        pred = self.ff(output)
        # [b, la, d_vocab]

        return pred

    def do_train(self, Y, a_ids, a_masks, cur_step, max_step):
        # Y: [b, nc*lc, d_hid]
        # a_ids: [b, la]
        # a_masks: [b, la]

        bz, la = a_ids.size()

        input_embds = self.embd_layer.encode_ans(a_ids=a_ids, a_masks=a_masks)
        outputs = self(Y=Y, a_embds=input_embds, a_masks=a_masks)
        # [b, la, d_vocab]
        outputs = outputs.softmax(dim=-1)

        input_embds = torch.zeros((bz, la, self.d_hid), dtype=Y.dtype, device=Y.device)
        input_embds[bz:, 0] = self.embd_layer.encode_tok(
            ids2dist(torch.full((1,), self.cls_tok_id, device=Y.device), self.d_vocab)
        ).squeeze(1)
        for b in range(bz):
            for i in range(1, la - 1):
                if is_schedsampl(cur_step, max_step):
                    input_embds[b, i] = self.embd_layer.encode_tok(outputs[b, i - 1])
                else:
                    tok = ids2dist(a_ids[b, i], self.d_vocab)
                    input_embds[b, i] = self.embd_layer.encode_tok(tok)
            input_embds[bz:, 0] = self.embd_layer.encode_tok(
                ids2dist(torch.full((1,), self.sep_tok_id, device=Y.device), self.d_vocab)
            ).squeeze(1)

        output_mle = self(Y=Y, a_embds=input_embds, a_masks=a_masks)

        loss = self.get_loss(output_mle, a_ids, a_masks)

        return loss, output_mle

    def do_predict(self, Y, a_ids, a_masks):
        # Y: [b, nc*lc, d_hid]

        bz = Y.size(0)

        ## Init input_embs with cls embedding
        cls_ids = torch.full(
            (bz,),
            fill_value=self.cls_tok_id,
            device=Y.device,
            requires_grad=False,
        )
        input_ids = [cls_ids.unsqueeze(1)]

        for ith in range(1, self.la + 1):
            input_embds = self.embd_layer.encode_ans(a_ids=torch.cat(input_ids, dim=-1))
            outputs = self(Y=Y, a_embds=input_embds)
            # [b, ith, d_vocab]

            if ith == self.la:
                break

            _, topi = torch.topk(torch.softmax(outputs[:, -1, :], dim=-1), k=1)

            input_ids.append(topi.detach())

        output_mle = outputs
        # [b, la, d_vocab]

        loss = self.get_loss(output_mle, a_ids, a_masks)

        return loss, output_mle
