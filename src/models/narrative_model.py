from typing import Any, Optional
import json, re


from transformers import AdamW, BertTokenizer
import pytorch_lightning as plt
import torch.nn.functional as torch_F
import torch.nn as torch_nn
import torch

from src.datamodules.narrative_datamodule import NarrativeDataModule
from src.models.layers.reasoning_layer.memorygraph_layer import GraphBasedMemoryLayer
from src.models.layers.finegrain_layer import FineGrain
from src.models.layers.ans_infer_layer import BertDecoder
from src.models.utils import BeamSearchHuggingface, BeamSearchOwn
from utils.utils import ipot, get_scores

EPSILON = 10e-10


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        l_q,
        l_c,
        l_a,
        n_nodes,
        n_edges,
        n_gru_layers,
        d_hid,
        d_bert,
        d_vocab,
        d_graph,
        lr,
        w_decay,
        switch_frequency,
        beam_size,
        n_gram_beam,
        temperature,
        topP,
        path_bert,
        path_pred,
        path_train_pred,
        path_valid_pred,
        datamodule: NarrativeDataModule = None,
        **kwargs
    ):

        super().__init__()

        self.d_vocab = d_vocab
        self.lr = lr
        self.w_decay = w_decay
        self.beam_size = beam_size
        self.n_gram_beam = n_gram_beam
        self.temperature = temperature
        self.topP = topP
        self.l_a = l_a
        self.switch_frequency = switch_frequency

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_bert)
        self.datamodule = datamodule

        self.path_pred = path_pred
        self.path_train_pred = path_train_pred
        self.path_valid_pred = path_valid_pred

        #############################
        # Define model
        #############################
        self.embd_layer = FineGrain(
            l_c,
            n_gru_layers,
            d_bert,
            path_bert,
        )
        self.reasoning = GraphBasedMemoryLayer(
            batch_size,
            l_q,
            l_a,
            d_hid,
            d_bert,
            d_graph,
            n_nodes,
            n_edges,
        )
        self.ans_infer = BertDecoder(
            l_a=l_a,
            d_bert=d_bert,
            d_vocab=d_vocab,
            cls_tok_id=self.bert_tokenizer.cls_token_id,
            embd_layer=self.embd_layer,
        )

        ## Freeeze some parameters
        list_freeze_sets = [
            self.embd_layer.bert_emb.parameters(),
            # self.ans_infer.decoder.parameters(),
        ]
        for params in list_freeze_sets:
            for param in params:
                param.requires_grad = False

        #############################
        # Define things
        #############################
        self.criterion = torch_nn.CrossEntropyLoss()

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def get_score_from_outputs(self, outputs):
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs:
            try:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = get_scores(**pair)
            except ValueError:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = 0, 0, 0, 0

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

            n_samples += 1

        return (
            bleu_1 / n_samples,
            bleu_4 / n_samples,
            meteor / n_samples,
            rouge_l / n_samples,
        )

    def get_prediction(self, output_mle, a1_ids, a2_ids):
        _, prediction = torch.topk(torch_F.log_softmax(output_mle, dim=1), 1, dim=1)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(prediction.squeeze(1), a1_ids, a2_ids)
        ]

        return prediction

    def get_loss(self, output_mle, output_ot, a_ids, a_masks, gamma=0.05):
        # output_mle: [b, d_vocab, l_a-1]
        # output_ot: [b, l_a-1, d_hid]
        # a_ids: [b, l_a-1]
        # a_masks: [b, l_a-1]

        # Calculate MLE loss
        loss_mle = self.criterion(output_mle, a_ids)

        # Calculate OT loss
        ans = self.embd_layer.encode_ans(input_ids=a_ids, input_masks=a_masks)
        # [b, l_a-1, d_hid]

        loss_ot = ipot(output_ot, ans)

        total_loss = loss_mle + gamma * loss_ot

        return total_loss

    def model(
        self,
        ques_ids,
        ques_mask,
        ans_ids,
        ans_mask,
        context_ids,
        context_mask,
        cur_step,
        max_step,
        is_valid: bool = False,
    ):
        # ques       : [b, l_q]
        # ques_mask  : [b, l_q]
        # context_ids  : [b, n_paras, l_c]
        # context_mask : [b, n_paras, l_c]
        # ans_ids: [b, l_a]
        # ans_mask   : [b, l_a]

        ####################
        # Embed question, context and answer
        ####################
        ques, context = self.embd_layer.encode_ques_para(
            ques_ids=ques_ids,
            context_ids=context_ids,
            ques_mask=ques_mask,
            context_mask=context_mask,
        )
        # ques : [b, l_q, d_bert]
        # context: [b, n_paras, d_bert]
        # ans  : [b, l_a, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, context)
        # [b, l_a, d_bert]

        ####################
        # Generate answer
        ####################
        return self.ans_infer.do_train(
            Y=Y,
            ans_ids=ans_ids,
            ans_mask=ans_mask,
            cur_step=cur_step,
            max_step=max_step,
            is_valid=is_valid,
        )

    def training_step(self, batch: Any, batch_idx: int):
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        ans1_mask = batch["ans1_mask"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        output_mle, output_ot = self.model(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            ans_ids=ans1_ids,
            ans_mask=ans1_mask,
            context_ids=context_ids,
            context_mask=context_mask,
            cur_step=batch_idx,
            max_step=self.datamodule.data_train.size_dataset
            // self.datamodule.batch_size,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        loss = self.get_loss(output_mle, output_ot, ans1_ids[:, 1:], ans1_mask[:, 1:])

        prediction = self.get_prediction(output_mle, ans1_ids, ans2_ids)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/sampling",
            self.ans_infer.t,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        return {"loss": loss, "pred": prediction}

    def training_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(p)

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = self.get_score_from_outputs(preds)

        self.log("train/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("train/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("train/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("train/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans1_mask = batch["ans1_mask"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        pred = self.model(
            ques_ids, ques_mask, ans1_ids, ans1_mask, context_ids, context_mask
        )
        # [b, d_vocab, l_a]

        loss = self.criterion(pred[:, :, :-1], ans1_ids[:, 1:])

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        ans1_mask = batch["ans1_mask"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        output_mle, output_ot = self.model(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            ans_ids=ans1_ids,
            ans_mask=ans1_mask,
            context_ids=context_ids,
            context_mask=context_mask,
            is_valid=True,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        loss = self.get_loss(output_mle, output_ot, ans1_ids[:, 1:], ans1_mask[:, 1:])

        prediction = self.get_prediction(output_mle, ans1_ids, ans2_ids)

        return {"loss": loss, "pred": prediction}

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        path = "/home/ubuntu/NarrativeGraph/data/valid_prediction.json"
        with open(path, "a+") as pred_file:
            json.dump(outputs["pred"], pred_file, indent=2, ensure_ascii=False)

        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs["pred"]:
            bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_scores(**pair)

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

            n_samples += 1

        self.log("valid/bleu_1", bleu_1 / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4 / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l / n_samples, on_epoch=True, prog_bar=False)

    def validation_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(p)

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = self.get_score_from_outputs(preds)

        self.log("train/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("train/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("train/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("train/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(), lr=self.lr, weight_decay=self.w_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=15, eta_min=0
            ),
        }

    def on_validation_end(self) -> None:
        if self.current_epoch % self.switch_frequency == 0 and self.current_epoch != 0:
            self.datamodule.switch_answerability()

    #########################################
    # FOR PREDICTION PURPOSE
    #########################################
    def forward(self, ques_ids, ques_mask, context_ids, context_mask):
        # ques_ids       : [b, l_q]
        # ques_mask  : [b, l_q]
        # context_ids      : [b, n_paras, l_c]
        # context_mask : [b, n_paras, l_c]

        b = ques_ids.size(0)

        ####################
        # Embed question, context
        ####################
        ques, context = self.embd_layer.encode_ques_para(
            ques_ids=ques_ids,
            context_ids=context_ids,
            ques_mask=ques_mask,
            context_mask=context_mask,
        )
        # ques : [b, l_q, d_bert]
        # context: [b, n_paras, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, context)
        # [b, l_a, d_bert]

        ####################
        # Generate answer
        ####################
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

        outputs = []

        beam_search = BeamSearchOwn(
            beam_size=self.beam_size,
            init_tok=self.bert_tokenizer.cls_token_id,
            stop_tok=self.bert_tokenizer.sep_token_id,
            max_len=self.len,
            model=self.generate_own,
            no_repeat_ngram_size=self.n_gram_beam,
            topk_strategy="select_mix_beam",
        )

        for b_ in range(b):
            indices = beam_search.search(Y[b_, :, :])

            outputs.append(indices)

        outputs = torch.tensor(outputs, device=self.device, dtype=torch.long)

        return outputs

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

    def generate_own(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [seq_len<=200]
        # encoder_outputs  : [l_a, d_bert]

        decoder_input_ids = (
            torch.LongTensor(decoder_input_ids)
            .type_as(encoder_outputs)
            .long()
            .unsqueeze(0)
        )

        decoder_input_mask = torch.ones(decoder_input_ids.shape, device=self.device)
        decoder_input_embd = self.embd_layer.encode_ans(
            decoder_input_ids, decoder_input_mask
        )
        # [1, seq=*, d_bert]

        encoder_outputs = encoder_outputs.unsqueeze(0)

        output = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
        # [1, seq=*, d_vocab]

        output = output.squeeze(0)
        # [seq=*, d_vocab]

        return output

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int],
    ) -> Any:
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        prediction = self(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            context_ids=context_ids,
            context_mask=context_mask,
        )

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(prediction.squeeze(1), ans1_ids, ans2_ids)
        ]

        return prediction

    def on_predict_batch_end(
        self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        #######################
        # Calculate metrics
        #######################
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs:
            try:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_scores(**pair)
            except ValueError:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = 0, 0, 0, 0

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

            n_samples += 1

        #######################
        # Log prediction and metrics
        #######################
        with open(self.path_pred, "a+") as pred_file:
            json.dump(
                {
                    "metrics": {
                        "bleu_1": bleu_1 / n_samples,
                        "bleu_4": bleu_4 / n_samples,
                        "meteor": meteor / n_samples,
                        "rouge_l": rouge_l / n_samples,
                    },
                    "predictions": outputs,
                },
                pred_file,
                indent=2,
                ensure_ascii=False,
            )
