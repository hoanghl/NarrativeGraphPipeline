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

        # ## Freeeze some parameters
        # list_freeze_sets = [
        #     # self.embd_layer.bert_emb.parameters(),
        #     # self.ans_infer.decoder.parameters(),
        # ]
        # for params in list_freeze_sets:
        #     for param in params:
        #         param.requires_grad = False

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

    def get_prediction(self, pairs):
        pairs = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred)),
                "trg": " ".join(self.bert_tokenizer.convert_ids_to_tokens(trg)),
            }
            for pred, trg in zip(pairs["pred"], pairs["trg"])
        ]

        return pairs

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
        # logist: list of [b, la, d_vocab]

        if not self.is_tuning:
            logist = [torch.argmax(logist_, dim=-1) for logist_ in logist]
            trgs_ = [batch["a1_ids"], batch["a2_ids"]] if self.use_2_answers else [batch["a1_ids"]]

            bz = batch["q_ids"].size(0)
            preds, trgs = [], []
            for i in range(bz):
                for output, trg in zip(logist, trgs_):
                    preds.append(output[i])
                    trgs.append(trg[i])

            return {
                "loss": loss,
                "pred": preds,
                "trg": trgs,
                "size_pred": logist[0].size(-1),
                "size_trg": trgs_[0].size(-1),
            }
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        if self.is_tuning is True:
            return {"loss": batch_parts["loss"].mean()}

        size_pred = (
            batch_parts["size_pred"]
            if isinstance(batch_parts["size_pred"], int)
            else batch_parts["size_pred"][0]
        )
        size_trg = (
            batch_parts["size_trg"]
            if isinstance(batch_parts["size_trg"], int)
            else batch_parts["size_trg"][0]
        )

        preds, trgs = [], []
        for pred, trg in zip(batch_parts["pred"], batch_parts["trg"]):
            preds.extend(pred.view(-1, size_pred).cpu().detach())
            trgs.extend(trg.view(-1, size_trg).cpu().detach())

        loss = batch_parts["loss"].mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        return {"loss": loss, "pred": preds, "trg": trgs}

    def training_epoch_end(self, outputs) -> None:
        if self.is_tuning is True:
            return

        pairs = {"pred": [], "trg": []}
        loss = 0
        for output in outputs:
            pairs["pred"].extend(output["pred"])
            pairs["trg"].extend(output["trg"])
            loss += output["loss"]
        outputs = pairs

        ## Calculate B-1, B-4, METEOR and ROUGE-L
        outputs = self.get_prediction(outputs)

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

        self.log("train/bleu_1", bleu_1)
        self.log("train/bleu_4", bleu_4)
        self.log("train/meteor", meteor)
        self.log("train/rouge_l", rouge_l)

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

        return {
            "pred": preds,
            "trg": trgs,
            "size_pred": logist[0].size(-1),
            "size_trg": trgs_[0].size(-1),
        }

    def validation_step_end(self, batch_parts):
        size_pred = (
            batch_parts["size_pred"]
            if isinstance(batch_parts["size_pred"], int)
            else batch_parts["size_pred"][0]
        )
        size_trg = (
            batch_parts["size_trg"]
            if isinstance(batch_parts["size_trg"], int)
            else batch_parts["size_trg"][0]
        )

        preds, trgs = [], []
        for pred, trg in zip(batch_parts["pred"], batch_parts["trg"]):
            preds.extend(pred.view(-1, size_pred).cpu().detach())
            trgs.extend(trg.view(-1, size_trg).cpu().detach())

        return {"pred": preds, "trg": trgs}

    def validation_epoch_end(self, outputs) -> None:
        pairs = {"pred": [], "trg": []}
        for output in outputs:
            pairs["pred"].extend(output["pred"])
            pairs["trg"].extend(output["trg"])
        outputs = pairs

        outputs = self.get_prediction(outputs)

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

        self.log("valid/bleu_1", bleu_1)
        self.log("valid/bleu_4", bleu_4)
        self.log("valid/meteor", meteor)
        self.log("valid/rouge_l", rouge_l)

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

        return {
            "pred": preds,
            "trg": trgs,
            "size_pred": logist[0].size(-1),
            "size_trg": trgs_[0].size(-1),
        }

    def test_step_end(self, batch_parts):
        size_pred = (
            batch_parts["size_pred"]
            if isinstance(batch_parts["size_pred"], int)
            else batch_parts["size_pred"][0]
        )
        size_trg = (
            batch_parts["size_trg"]
            if isinstance(batch_parts["size_trg"], int)
            else batch_parts["size_trg"][0]
        )

        preds, trgs = [], []
        for pred, trg in zip(batch_parts["pred"], batch_parts["trg"]):
            preds.extend(pred.view(-1, size_pred).cpu().detach())
            trgs.extend(trg.view(-1, size_trg).cpu().detach())

        return {"pred": preds, "trg": trgs}

    def test_epoch_end(self, outputs) -> None:
        pairs = {"pred": [], "trg": []}
        for output in outputs:
            pairs["pred"].extend(output["pred"])
            pairs["trg"].extend(output["trg"])
        outputs = pairs

        outputs = self.get_prediction(outputs)

        if not self.is_tuning:
            with open(self.path_test_pred, "a+") as pred_file:
                json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

        self.log("bleu_1", bleu_1)
        self.log("bleu_4", bleu_4)
        self.log("meteor", meteor)
        self.log("rouge_l", rouge_l)
