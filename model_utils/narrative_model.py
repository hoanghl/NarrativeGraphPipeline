from typing import Any, Optional
import json

from transformers import AdamW, BertTokenizer
import pytorch_lightning as plt
import torch.nn.functional as torch_F
import torch.nn as torch_nn
import torch


from data_utils.narrative_datamodule import NarrativeDataModule
from model_utils.layers.reasoning_layer import Reasoning
from model_utils.layers.bertbasedembd_layer import BertBasedEmbedding
from model_utils.layers.ans_infer_layer import Decoder
from utils.generator import GeneratorOwn
from utils.utils import ipot, get_scores

EPSILON = 10e-10


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        l_q,
        l_c,
        l_a,
        n_heads,
        d_hid,
        d_bert,
        d_vocab,
        lr,
        w_decay,
        dropout,
        beam_size,
        n_gram_beam,
        path_bert,
        path_pred,
        path_train_pred,
        path_valid_pred,
        datamodule,
        **kwargs
    ):

        super().__init__()

        self.d_vocab = d_vocab
        self.lr = lr
        self.w_decay = w_decay
        self.beam_size = beam_size
        self.n_gram_beam = n_gram_beam
        self.l_a = l_a

        self.path_pred = path_pred
        self.path_train_pred = path_train_pred
        self.path_valid_pred = path_valid_pred

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_bert)
        self.datamodule: NarrativeDataModule = datamodule

        #############################
        # Define model
        #############################
        self.embd_layer = BertBasedEmbedding(
            d_bert=d_bert, d_hid=d_hid, path_bert=path_bert
        )
        self.reasoning = Reasoning(
            l_q=l_q,
            l_c=l_c,
            n_heads=n_heads,
            d_hid=d_hid,
            dropout=dropout,
            device=self.device,
        )
        self.ans_infer = Decoder(
            l_a=l_a,
            d_vocab=d_vocab,
            d_hid=d_hid,
            tokenizer=self.bert_tokenizer,
            embd_layer=self.embd_layer,
        )

        ## Freeeze some parameters
        list_freeze_sets = [
            self.embd_layer.bert_emb.parameters(),
            # self.ans_infer.decoder.parameters(),
        ]
        for params in list_freeze_sets:
            for param in params:
                param.requires_grad = True

        #############################
        # Define things
        #############################
        self.criterion = torch_nn.CrossEntropyLoss()

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def get_loss(self, output_mle, output_ot, a_ids, a_masks, gamma=0.1):
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

    def model(
        self,
        q_ids,
        q_masks,
        a_ids,
        a_masks,
        c_ids,
        c_masks,
        cur_step=0,
        max_step=0,
        is_valid: bool = False,
    ):
        # q_ids: [b, l_q]
        # q_masks: [b, l_q]
        # c_ids: [b, n_c, l_c]
        # c_masks: [b, n_c, l_c]
        # a_ids: [b, l_a]
        # a_masks: [b, l_a]

        ####################
        # Embed question, c and answer
        ####################
        q, c = self.embd_layer.encode_ques_para(
            q_ids=q_ids,
            c_ids=c_ids,
            q_masks=q_masks,
            c_masks=c_masks,
        )
        # q : [b, l_q, d_hid]
        # c: [b, n_c, l_c, d_hid]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(q=q, c=c)
        # [b, 1, d_hid]

        ####################
        # Generate answer
        ####################

        return (
            self.ans_infer.do_predict(
                Y=Y,
                a_masks=a_masks,
            )
            if is_valid
            else self.ans_infer.do_train(
                Y=Y,
                a_ids=a_ids,
                a_masks=a_masks,
                cur_step=cur_step,
                max_step=max_step,
            )
        )
        # pred: [b, d_vocab, l_a - 1]

    def training_step(self, batch: Any, batch_idx):
        q_ids = batch["q_ids"]
        q_masks = batch["q_masks"]
        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]
        a1_masks = batch["a1_masks"]
        c_ids = batch["c_ids"]
        c_masks = batch["c_masks"]

        output_mle, output_ot = self.model(
            q_ids=q_ids,
            q_masks=q_masks,
            a_ids=a1_ids,
            a_masks=a1_masks,
            c_ids=c_ids,
            c_masks=c_masks,
            cur_step=batch_idx,
            max_step=self.datamodule.data_train.size_dataset
            // self.datamodule.batch_size,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        loss = self.get_loss(output_mle, output_ot, a1_ids[:, 1:], a1_masks[:, 1:])
        prediction = self.get_prediction(output_mle, a1_ids, a2_ids)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/t", self.ans_infer.t, on_step=True, on_epoch=False, prog_bar=False
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

    def test_step(self, batch: Any, batch_idx):
        q_ids = batch["q_ids"]
        q_masks = batch["q_masks"]
        a1_ids = batch["a1_ids"]
        a1_masks = batch["a1_masks"]
        c_ids = batch["c_ids"]
        c_masks = batch["c_masks"]

        output_mle, output_ot = self.model(
            q_ids=q_ids,
            q_masks=q_masks,
            a_ids=a1_ids,
            a_masks=a1_masks,
            c_ids=c_ids,
            c_masks=c_masks,
            cur_step=batch_idx,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        loss = self.get_loss(output_mle, output_ot, a1_ids[:, 1:], a1_masks[:, 1:])

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx):
        q_ids = batch["q_ids"]
        q_masks = batch["q_masks"]
        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]
        a1_masks = batch["a1_masks"]
        c_ids = batch["c_ids"]
        c_masks = batch["c_masks"]

        output_mle, output_ot = self.model(
            q_ids=q_ids,
            q_masks=q_masks,
            a_ids=a1_ids,
            a_masks=a1_masks,
            c_ids=c_ids,
            c_masks=c_masks,
            is_valid=True,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        loss = self.get_loss(output_mle, output_ot, a1_ids[:, 1:], a1_masks[:, 1:])
        prediction = self.get_prediction(output_mle, a1_ids, a2_ids)

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "pred": prediction}

    def validation_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(p)

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = self.get_score_from_outputs(preds)
        self.log("valid/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        # TODO: Find the way to finish
        if self.current_epoch % 4 == 0 and self.current_epoch != 0:
            self.datamodule.switch_answerability()

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(), lr=self.lr, weight_decay=self.w_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=16, eta_min=0
            ),
        }

    #########################################
    # FOR PREDICTION PURPOSE
    #########################################
    def forward(
        self,
        q_ids,
        q_masks,
        c_ids,
        c_masks,
    ):
        # q_ids: [b, l_q]
        # q_masks: [b, l_q]
        # c_ids: [b, n_c, l_c]
        # c_masks: [b, n_c, l_c]

        b = q_masks.size()[0]

        ####################
        # Embed question, c and answer
        ####################
        q, c = self.embd_layer.encode_ques_para(
            q_ids=q_ids,
            c_ids=c_ids,
            q_masks=q_masks,
            c_masks=c_masks,
        )
        # q : [b, l_q, d_hid]
        # c: [b, n_c, l_c, d_hid]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(q=q, c=c)
        # [b, 1, d_hid]

        ####################
        # Generate answer
        ####################
        outputs = []

        generator = GeneratorOwn(
            beam_size=self.beam_size,
            init_tok=self.bert_tokenizer.cls_token_id,
            stop_tok=self.bert_tokenizer.sep_token_id,
            max_len=self.l_a,
            model=self.generate,
            no_repeat_ngram_size=self.n_gram_beam,
            topk_strategy="topk",
        )

        for b_ in range(b):
            indices = generator.search(Y[b_, :, :])

            outputs.append(indices)

        outputs = torch.LongTensor(outputs, device=self.device)

        return outputs

    def generate(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [list: len_]
        # encoder_outputs  : [l_c, d_hid]

        decoder_input_ids = (
            torch.LongTensor(decoder_input_ids)
            .type_as(encoder_outputs)
            .long()
            .unsqueeze(0)
        )

        decoder_input_mask = torch.ones(decoder_input_ids.shape, device=self.device)
        decoder_input_embd = self.embd_layer.encode_ans(
            input_ids=decoder_input_ids, input_masks=decoder_input_mask
        )
        # [1, len_, d_bert]

        encoder_outputs = encoder_outputs.unsqueeze(0)

        output = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
        # [1, len_, d_vocab]

        output = output.squeeze(0)
        # [len_, d_vocab]

        return output

    def predict_step(
        self,
        batch: Any,
        batch_idx,
        dataloader_idx: Optional[int],
    ) -> Any:
        q_ids = batch["q_ids"]
        q_masks = batch["q_masks"]
        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]
        c_ids = batch["c_ids"]
        c_masks = batch["c_masks"]

        pred = self(
            q_ids=q_ids,
            q_masks=q_masks,
            c_ids=c_ids,
            c_masks=c_masks,
        )
        # pred: [b, l_a, d_vocab]

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(pred.squeeze(1), a1_ids, a2_ids)
        ]

        return prediction

    def on_predict_batch_end(
        self, outputs: Optional[Any], batch: Any, batch_idx, dataloader_idx
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
