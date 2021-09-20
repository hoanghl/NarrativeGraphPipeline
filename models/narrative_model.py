import json
import os
from typing import Any, Optional

import pytorch_lightning as plt
import torch
import torch.nn as torch_nn
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
from utils.model_utils import get_scores

from models.Backbone import Backbone


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        lq,
        lc,
        la,
        d_bert,
        d_hid,
        d_vocab,
        num_heads,
        num_heads_persistent,
        num_layers_p,
        num_layers_q,
        num_layers_a,
        lr,
        w_decay,
        dropout,
        size_dataset_train,
        max_epochs,
        warmup_rate,
        path_pretrained,
        path_test_pred,
        path_train_pred,
        path_valid_pred,
        use_2_answers,
        is_tuning=False,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.d_vocab = d_vocab
        self.la = la
        self.lr = lr
        self.w_decay = w_decay
        self.warmup_rate = warmup_rate
        self.size_dataset_train = size_dataset_train
        self.max_epochs = max_epochs
        self.path_test_pred = path_test_pred
        self.path_train_pred = path_train_pred
        self.path_valid_pred = path_valid_pred
        self.use_2_answers = use_2_answers
        self.is_tuning = is_tuning
        self.bert_tokenizer = BertTokenizer.from_pretrained(path_pretrained)

        os.makedirs(os.path.dirname(path_valid_pred), exist_ok=True)

        #############################
        # Define model
        #############################

        self.model = Backbone(
            lq=lq,
            lc=lc,
            d_bert=d_bert,
            d_hid=d_hid,
            d_vocab=d_vocab,
            num_heads=num_heads,
            num_heads_persistent=num_heads_persistent,
            num_layers_p=num_layers_p,
            num_layers_q=num_layers_q,
            num_layers_a=num_layers_a,
            dropout=dropout,
            path_pretrained=path_pretrained,
            criterion=torch_nn.CrossEntropyLoss(ignore_index=self.bert_tokenizer.pad_token_id),
            device=self.device,
        )

        self.val_results = []
        self.train_results = []
        self.test_results = []

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
                "weight_decay": self.w_decay,
            },
            {
                "params": params_nodecay,
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.lr)

        n_training_steps = self.size_dataset_train // self.batch_size * self.max_epochs
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(n_training_steps * self.warmup_rate),
                num_training_steps=n_training_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def training_step(self, batch: Any, batch_idx: int):
        loss, logist = self.model.do_train(
            q=batch["q_ids"],
            c=batch["c_ids"],
            a1=batch["a1_ids"],
            a2=batch["a2_ids"],
            a1_mask=batch["a1_masks"],
            a2_mask=batch["a2_masks"],
            use_2_ans=self.use_2_answers,
        )
        # logist: list of [b, la + 2, d_vocab]

        if self.is_tuning is True:
            return {"loss": loss}

        logist = [torch.argmax(logist_, dim=-1) for logist_ in logist]
        trgs_ = [batch["a1_ids"], batch["a2_ids"]] if self.use_2_answers else [batch["a1_ids"]]

        bz = batch["q_ids"].size(0)
        preds, trgs = [], []
        for i in range(bz):
            for output, trg in zip(logist, trgs_):
                preds.append(output[i])
                trgs.append(trg[i])
                self.train_results.append(
                    {
                        "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(output[i])),
                        "trg": " ".join(self.bert_tokenizer.convert_ids_to_tokens(trg[i])),
                    }
                )

        return {"loss": loss}

    def training_step_end(self, batch_parts):
        loss = batch_parts["loss"].mean()

        if not self.is_tuning:
            self.log("train/loss", loss, on_step=True, on_epoch=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        if self.is_tuning is True:
            return

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(self.train_results, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(self.train_results)

        self.log("train/bleu_1", bleu_1)
        self.log("train/bleu_4", bleu_4)
        self.log("train/meteor", meteor)
        self.log("train/rouge_l", rouge_l)

        self.train_results = []

    def validation_step(self, batch: Any, batch_idx):
        logist = self.model.do_predict(batch["q_ids"], batch["c_ids"], self.la)
        # logist: [b, la]

        logist = [logist, logist] if self.use_2_answers else [logist]
        trgs_ = [batch["a1_ids"], batch["a2_ids"]] if self.use_2_answers else [batch["a1_ids"]]

        bz = batch["q_ids"].size(0)
        preds, trgs = [], []
        for i in range(bz):
            for output, trg in zip(logist, trgs_):
                preds.append(output[i])
                trgs.append(trg[i])
                self.val_results.append(
                    {
                        "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(output[i])),
                        "trg": " ".join(self.bert_tokenizer.convert_ids_to_tokens(trg[i])),
                    }
                )

    def validation_epoch_end(self, outputs) -> None:
        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(self.val_results, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(self.val_results)

        self.log("valid/bleu_1", bleu_1)
        self.log("valid/bleu_4", bleu_4)
        self.log("valid/meteor", meteor)
        self.log("valid/rouge_l", rouge_l)

        self.val_results = []

    def test_step(self, batch, batch_idx):
        logist = self.model.do_predict(batch["q_ids"], batch["c_ids"], self.la)
        # logist: [b, la]

        logist = [logist, logist] if self.use_2_answers else [logist]
        trgs_ = [batch["a1_ids"], batch["a2_ids"]] if self.use_2_answers else [batch["a1_ids"]]

        bz = batch["q_ids"].size(0)
        preds, trgs = [], []
        for i in range(bz):
            for output, trg in zip(logist, trgs_):
                preds.append(output[i])
                trgs.append(trg[i])
                self.test_results.append(
                    {
                        "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(output[i])),
                        "trg": " ".join(self.bert_tokenizer.convert_ids_to_tokens(trg[i])),
                    }
                )

    def test_epoch_end(self, outputs) -> None:
        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(self.test_results, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(self.test_results)

        self.log("bleu_1", bleu_1)
        self.log("bleu_4", bleu_4)
        self.log("meteor", meteor)
        self.log("rouge_l", rouge_l)

        self.test_results = []
