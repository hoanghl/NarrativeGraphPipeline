import numpy as np
import torch
import torch.nn as torch_nn


class Decoder(torch_nn.Module):
    def __init__(self, l_a, d_bert, d_vocab, d_hid, dropout, tokenizer, embd_layer):
        super().__init__()

        self.tau = 0.2
        self.l_a = l_a

        self.tokenizer = tokenizer
        self.embd_layer = embd_layer
        self.lstm = torch_nn.LSTM(d_hid, d_hid, 2, batch_first=True, dropout=dropout)
        self.lin0 = torch_nn.Linear(d_bert, d_hid, bias=False)
        self.lin1 = torch_nn.Linear(d_hid, d_hid)
        self.lin2 = torch_nn.Linear(d_hid, d_hid)
        self.lin4 = torch_nn.Linear(d_hid, 1)
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid), torch_nn.Dropout(dropout), torch_nn.Tanh()
        )
        self.lin5 = torch_nn.Sequential(
            torch_nn.Linear(d_hid * 2, d_hid * 2), torch_nn.Linear(d_hid * 2, d_vocab)
        )
        self.lin_swch_1 = torch_nn.Linear(d_hid * 2, 1)
        self.lin_swch_2 = torch_nn.Linear(d_hid, 1)
        self.lin_swch_3 = torch_nn.Linear(d_hid, 1)
        self.lin6 = torch_nn.Linear(d_hid, d_hid)
        self.lin7 = torch_nn.Linear(d_hid, d_hid)
        self.ff2 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid), torch_nn.Dropout(dropout), torch_nn.Tanh()
        )

    def attentive_q(self, output, q):
        # output: [b, d_hid]
        # q: [b, l_q, d_hid]

        e_t = self.ff2(self.lin6(q) + self.lin7(output.unsqueeze(1)))
        # [b, l_q, d_hid]
        e_t = self.lin4(e_t)
        # [b, l_q, 1]

        h = (torch.softmax(e_t, dim=1) * q).sum(dim=1, keepdim=True)

        return h

    def attention(self, Y, output, q):
        # Y: [b, n_c*l_c, d_hid]
        # output: [b, d_hid]
        # q: [b, l_q, d_hid]

        a1 = self.lin1(Y)
        a2 = self.lin2(output.unsqueeze(1))
        a3 = self.attentive_q(output, q)
        e_t = self.ff1(a1 + a2 + a3)
        # [b, n_c*l_c, d_hid]
        e_t = self.lin4(e_t)
        # [b, n_c*l_c, 1]

        extract_dist = torch.softmax(e_t, dim=1)

        return extract_dist

    def pointer_generator(self, Y, c_ids, q, output, a_t):
        # Y: [b, n_c*l_c, d_hid * 4]
        # c_ids: [b, n_c*l_c]
        # q: [b, l_q, d_hid]
        # output: [b, d_hid]
        # a_t: [b, d_hid]

        #########################
        ## Calculate extractive dist over context
        #########################
        extract_dist = self.attention(Y, output, q)
        # [b, n_c*l_c, 1]

        #########################
        ## Calculate abstractive dist over gen vocab
        #########################
        contx = (extract_dist * Y).sum(dim=1)
        # [b, d_hid]
        contx = torch.cat((contx, output), dim=-1)
        # [b, d_hid * 2]
        abstr_dist = self.lin5(contx)
        # [b, d_vocab]

        #########################
        ## Calculate extract-abstract switch and combine them
        #########################
        switch = torch.sigmoid(
            self.lin_swch_1(contx) + self.lin_swch_2(a_t) + self.lin_swch_3(output)
        )
        # [b, 1]

        final = switch * abstr_dist
        # [b, d_vocab]

        # Scatter
        extract_dist = (1 - switch) * extract_dist.squeeze(-1)
        final = final.scatter_add_(dim=-1, index=c_ids, src=extract_dist)
        # [b, d_vocab]

        return final

    def choose_scheduled_sampling(
        self,
        output,
        a,
        cur_step,
        max_step,
    ):

        # output: [b, d_vocab]
        # a: [b, 1, d_bert]

        t = (
            np.random.binomial(1, cur_step / max_step)
            if max_step != 0
            else np.random.binomial(1, 1)
        )

        if t == 0:
            input_tok = torch.argmax(torch.log_softmax(output, dim=-1), dim=-1, keepdim=True)
            input_embd = self.get_tok_embd(input_tok)
            return input_embd

        return a
        # [b, 1, d_bert]

    def do_train(self, Y, a_ids, a_masks, c_ids, q, cur_step, max_step):
        # Y: [b, n_c*l_c, d_bert]
        # a_ids: [b, l_a]
        # a_masks: [b, l_a]
        # c_ids: [b, n_c*l_c]
        # q: [b, l_q, d_bert]

        b = Y.size(0)

        Y = self.lin0(Y)
        # [b, n_c_*l_c, d_hid]
        q = self.lin0(q)
        # [b, l_q, d_hid]

        a = self.embd_layer.encode_ans(a_ids, a_masks)
        # [b, l_a, d_bert]
        a = self.lin0(a)
        # [b, l_a, d_hid]

        input_embd, h_n = a[:, 0].unsqueeze(1), None
        outputs = []
        for t in range(1, self.l_a):
            output, h_n = self.lstm(input_embd, h_n)
            # output: [b, 1, d_hid]
            # h_n: tuple

            output = self.pointer_generator(Y, c_ids, q, output.squeeze(1), a[:, t])
            # [b, d_vocab]

            outputs.append(output.unsqueeze(-1))

            input_embd = self.choose_scheduled_sampling(
                output, a[:, t].unsqueeze(1), cur_step, max_step
            )
            # input_embd = a[:, t].unsqueeze(1)
            # [b, 1, d_hid]

        outputs = torch.cat(outputs, dim=-1)
        # [b, d_vocab, l_a - 1]

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(torch.softmax(outputs.transpose(1, 2), dim=-1))
        # [b, l_a - 1, d_hid]

        output_mle = outputs
        # [b, d_vocab, l_a - 1]

        return output_mle, output_ot

    def get_tok_embd(self, tok):
        # tok: [b, 1]
        return self.lin0(self.embd_layer.bert_emb.embeddings.word_embeddings(tok))
        # [b, 1, d_hid]

    def do_predict(self, Y, c_ids, q):
        # Y: [b, n_c*l_c, d_bert]
        # c_ids: [b, n_c*l_c]
        # q: [b, l_q, d_bert]

        b = Y.size(0)

        Y = self.lin0(Y)
        # [b, n_c_*l_c, d_hid]
        q = self.lin0(q)
        # [b, l_q, d_hid]

        h_n = None
        input_tok = torch.full(
            (b, 1), self.tokenizer.cls_token_id, device=Y.device, dtype=torch.long
        )

        outputs = []

        for t in range(1, self.l_a):
            input_embd = self.get_tok_embd(input_tok)
            # [b, 1, d_hid]

            output, h_n = self.lstm(input_embd, h_n)
            # output: [b, 1, D_hid]
            # h_n: tuple

            final = self.pointer_generator(Y, c_ids, q, output.squeeze(1), output.squeeze(1))
            # [b, d_vocab]

            outputs.append(final.unsqueeze(-1))

            input_tok = torch.argmax(torch.log_softmax(final, dim=-1), dim=-1, keepdim=True)
            # [b, 1]

        outputs = torch.cat(outputs, dim=-1)
        # [b, d_vocab, l_a - 1]

        ## Get output for OT
        output_ot = self.embd_layer.get_output_ot(torch.softmax(outputs.transpose(1, 2), dim=-1))
        # [b, l_a - 1, d_hid]

        output_mle = outputs
        # [b, d_vocab, l_a - 1]

        return output_mle, output_ot
