import json
import os
from typing import Any, Optional

import pytorch_lightning as plt
import torch
import torch.nn as torch_nn
from transformers import (
    AdamW,
    BertTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from utils.model_utils import get_scores, ipot

from models.layers.ans_infer_layer import Decoder
from models.layers.bertbasedembd_layer import BertBasedEmbedding
from models.layers.reasoning_layer import Reasoning


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        lq,
        lc,
        la,
        n_heads,
        d_hid,
        d_bert,
        d_vocab,
        lr,
        dropout,
        size_dataset_train,
        max_epochs,
        warmup_rate,
        path_pretrained,
        path_pred,
        path_train_pred,
        path_valid_pred,
    ):

        super().__init__()

        self.d_vocab = d_vocab
        self.lr = lr
        self.la = la
        self.batch_size = batch_size
        self.size_dataset_train = size_dataset_train
        self.max_epochs = max_epochs
        self.warmup_rate = warmup_rate
        self.path_pred = path_pred
        self.path_train_pred = path_train_pred
        self.path_valid_pred = path_valid_pred
        self.bert_tokenizer = BertTokenizer.from_pretrained(path_pretrained)

        os.makedirs(os.path.dirname(path_valid_pred), exist_ok=True)

        #############################
        # Define model
        #############################
        self.embd_layer = BertBasedEmbedding(
            d_bert=d_bert, d_hid=d_hid, path_pretrained=path_pretrained
        )
        self.reasoning = Reasoning(
            lq=lq,
            lc=lc,
            n_heads=n_heads,
            d_hid=d_hid,
            dropout=dropout,
            device=self.device,
        )
        self.ans_infer = Decoder(
            la=la,
            d_hid=d_hid,
            d_vocab=d_vocab,
            dropout=dropout,
            tokenizer=self.bert_tokenizer,
            embd_layer=self.embd_layer,
            criterion=torch_nn.CrossEntropyLoss(ignore_index=self.bert_tokenizer.pad_token_id),
        )

        # ## Freeeze some parameters
        # list_freeze_sets = [
        #     # self.embd_layer.bert_emb.parameters(),
        #     # self.ans_infer.decoder.parameters(),
        # ]
        # for params in list_freeze_sets:
        #     for param in params:
        #         param.requires_grad = False

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################
    def get_prediction(self, output, a1_ids, a2_ids):

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(output.squeeze(1), a1_ids, a2_ids)
        ]

        return prediction

    def forward(
        self,
        q_ids,
        q_masks,
        c_ids,
        c_masks,
        a1_ids=None,
        a1_masks=None,
        cur_step=0,
        max_step=0,
        is_predict=False,
        **kwargs
    ):
        # q_ids: [b, lq]
        # q_masks: [b, lq]
        # c_ids: [b, n_c, lc]
        # c_masks: [b, n_c, lc]
        # a_ids: [b, la]
        # a_masks: [b, la]

        ####################
        # Embed question, c and answer
        ####################
        q, c = self.embd_layer.encode_ques_para(
            q_ids=q_ids,
            c_ids=c_ids,
            q_masks=q_masks,
            c_masks=c_masks,
        )
        # q : [b, lq, d_hid]
        # c: [b, n_c, lc, d_hid]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(q=q, c=c)
        # [b, n_c*lc, d_hid]

        ####################
        # Generate answer
        ####################

        return (
            self.ans_infer.do_predict(Y=Y, a_ids=a1_ids, a_masks=a1_masks)
            if is_predict
            else self.ans_infer.do_train(
                Y=Y,
                a_ids=a1_ids,
                a_masks=a1_masks,
                cur_step=cur_step,
                max_step=max_step,
            )
        )
        # pred: [b, d_vocab, la - 1]

    def training_step(self, batch: Any, batch_idx: int):
        loss, logist = self(**batch, is_predict=True)
        # logist: [b, la, d_vocab]

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return {
            "loss": loss,
            "pred": (
                torch.argmax(logist, dim=-1).cpu().detach(),
                batch["a1_ids"].cpu().detach(),
                batch["a2_ids"].cpu().detach(),
            ),
        }

    def training_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(self.get_prediction(p[0], p[1], p[2]))

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(preds)

        self.log("train/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("train/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("train/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("train/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx):
        return 0

    def validation_step(self, batch: Any, batch_idx):
        loss, logist = self(**batch, is_predict=True)
        # logist: [b, la, d_vocab]

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {
            "loss": loss,
            "pred": (
                torch.argmax(logist, dim=-1).cpu().detach(),
                batch["a1_ids"].cpu().detach(),
                batch["a2_ids"].cpu().detach(),
            ),
        }

    def validation_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(self.get_prediction(p[0], p[1], p[2]))

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(preds)

        self.log("valid/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay, params_nodecay = [], []
        for model in [self.embd_layer, self.reasoning, self.ans_infer]:
            for n, p in model.named_parameters():
                if not any(nd in n for nd in no_decay):
                    params_decay.append(p)
                else:
                    params_nodecay.append(p)

        optimizer_grouped_parameters = [
            {
                "params": params_decay,
                "weight_decay": 0.95,
            },
            {
                "params": params_nodecay,
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.lr)

        n_training_steps = self.size_dataset_train // self.batch_size * self.max_epochs
        lr_scheduler = {
            "scheduler": get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(n_training_steps * self.warmup_rate),
                num_training_steps=n_training_steps,
                num_cycles=6,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def predict_step(
        self,
        batch: Any,
        batch_idx,
        dataloader_idx: Optional[int],
    ) -> Any:
        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]

        output_mle, _ = self(
            **batch,
            is_predict=True,
        )
        # [b, d_vocab, la - 1]

        return {
            "pred": (
                torch.argmax(output_mle, dim=1).cpu().detach(),
                a1_ids.cpu().detach(),
                a2_ids.cpu().detach(),
            ),
        }

    def on_predict_batch_end(
        self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(self.get_prediction(p[0], p[1], p[2]))

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = self.get_score_from_outputs(preds)

        self.log("predict/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("predict/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("predict/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("predict/rouge_l", rouge_l, on_epoch=True, prog_bar=False)
