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
from utils.model_utils import get_scores

from models.layers.chime import CHIME


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        lq,
        lc,
        la,
        n_heads,
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

        self.model = CHIME(
            lq=lq,
            lc=lc,
            d_bert=d_bert,
            n_heads=n_heads,
            d_vocab=d_vocab,
            dropout=dropout,
            criterion=torch_nn.CrossEntropyLoss(ignore_index=self.bert_tokenizer.pad_token_id),
            path_pretrained=path_pretrained,
            device=self.device,
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
    def get_prediction(self, pairs):
        pairs = [
            {
                "pred": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(p)) for p in pair["pred"]
                ],
                "trg": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(p)) for p in pair["trg"]
                ],
            }
            for pair in pairs
        ]

        return pairs

    def training_step(self, batch: Any, batch_idx: int):
        loss, logist = self.model.do_train(
            batch["q_ids"], batch["c_ids"], batch["a1_ids"], batch["a2_ids"], use_2_ans=False
        )
        # logist: list of [b, la, d_vocab]

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        # logist: list of [b, la, d_vocab]

        logist = [torch.argmax(logist_, dim=1) for logist_ in logist]
        trgs = [batch["a1_ids"], batch["a2_ids"]] if len(logist) > 1 else [batch["a1_ids"]]

        bz = batch["q_ids"].size(0)
        preds = [
            {
                "pred": [logit[i].tolist() for logit in logist],
                "trg": [trg[i].tolist() for trg in trgs],
            }
            for i in range(bz)
        ]

        return {"loss": loss, "pred": preds}

    def training_epoch_end(self, outputs) -> None:
        preds = []
        for pred in [out["pred"] for out in outputs]:
            preds.extend(self.get_prediction(pred))

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
        logist = self.model.do_predict(batch["q_ids"], batch["c_ids"], self.la)
        # logist: [b, la]

        logist = [logist, logist]
        trgs = [batch["a1_ids"], batch["a2_ids"]]

        bz = batch["q_ids"].size(0)
        preds = [
            {
                "pred": [logit[i].tolist() for logit in logist],
                "trg": [trg[i].tolist() for trg in trgs],
            }
            for i in range(bz)
        ]

        return {"pred": preds}

    def validation_epoch_end(self, outputs) -> None:
        preds = []
        for pred in [out["pred"] for out in outputs]:
            preds.extend(self.get_prediction(pred))

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(preds)

        self.log("valid/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay, params_nodecay = [], []

        for n, p in self.model.named_parameters():
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
