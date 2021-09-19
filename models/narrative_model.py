import json
import os

import pytorch_lightning as plt
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.datamodule_utils import Vocab
from utils.model_utils import get_scores

from models.Backbone import Backbone


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        lc,
        la,
        nc,
        d_embd,
        d_hid,
        d_vocab,
        num_layers,
        block,
        lr,
        w_decay,
        dropout,
        warmup_rate,
        max_epochs,
        size_dataset_train,
        path_pretrained,
        path_valid_pred,
        path_train_pred,
        path_test_pred,
        path_vocab,
        use_2_answers,
        is_tuning=False,
    ):
        super().__init__()

        self.la = la
        self.lr = lr
        self.w_decay = w_decay
        self.warmup_rate = warmup_rate
        self.path_valid_pred = path_valid_pred
        self.path_train_pred = path_train_pred
        self.path_test_pred = path_test_pred
        self.n_training_steps = size_dataset_train // batch_size * max_epochs
        self.use_2_answers = use_2_answers
        self.is_tuning = is_tuning
        self.vocab = Vocab(path_vocab)

        #############################
        # Define model
        #############################
        self.model = Backbone(
            batch_size=batch_size,
            la=la,
            lc=lc,
            nc=nc,
            d_embd=d_embd,
            d_hid=d_hid,
            d_vocab=d_vocab,
            num_layers=num_layers,
            block=block,
            dropout=dropout,
            path_pretrained=path_pretrained,
            path_vocab=path_vocab,
            criterion=nn.CrossEntropyLoss(ignore_index=self.vocab.pad_id),
            device=self.device,
        )

        os.makedirs(os.path.dirname(self.path_train_pred), exist_ok=True)

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

        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.n_training_steps * self.warmup_rate),
                num_training_steps=self.n_training_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def training_step(self, batch, batch_idx: int):
        loss, logist = self.model.do_train(**batch)
        # logist: list of [b, la, d_vocab + lc]

        if self.is_tuning is True:
            return {"loss": loss}

        logist = [torch.argmax(logist_, dim=-1) for logist_ in logist]
        oovs = batch["oovs"]
        trgs_ = [batch["trg1_txt"], batch["trg2_txt"]] if len(logist) > 1 else [batch["trg1_txt"]]

        bz = batch["q_ids"].size(0)
        preds_txt, trgs_txt = [], []
        for i in range(bz):
            for output, trg_txt in zip(logist, trgs_):
                pred_txt = self.vocab.pred2s(output[i], oovs[i])
                preds_txt.append(pred_txt)
                trgs_txt.append(trg_txt[i])

        return {"loss": loss, "preds_txt": preds_txt, "trgs_txt": trgs_txt}

    def training_step_end(self, batch_parts):
        if self.is_tuning is True:
            return {"loss": batch_parts["loss"].mean()}

        loss = batch_parts["loss"].mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        preds_txt, trgs_txt = [], []
        for pred, trg in zip(batch_parts["preds_txt"], batch_parts["trgs_txt"]):
            preds_txt.append(" ".join(pred))
            trgs_txt.append(trg)

        return {"loss": loss, "preds_txt": preds_txt, "trgs_txt": trgs_txt}

    def training_epoch_end(self, outputs) -> None:
        if self.is_tuning is True:
            return

        outputs = outputs[0]
        outputs = [
            {"pred": pred, "trg": trg}
            for pred, trg in zip(outputs["preds_txt"], outputs["trgs_txt"])
        ]

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

        self.log("train/bleu_1", bleu_1)
        self.log("train/bleu_4", bleu_4)
        self.log("train/meteor", meteor)
        self.log("train/rouge_l", rouge_l)

    def validation_step(self, batch, batch_idx):
        loss, logist = self.model.do_predict(**batch)
        # logist: [b, la]

        logist = [logist, logist] if self.use_2_answers else [logist]
        oovs = batch["oovs"]
        trgs_ = [batch["trg1_txt"], batch["trg2_txt"]] if len(logist) > 1 else [batch["trg1_txt"]]

        bz = batch["q_ids"].size(0)
        preds_txt, trgs_txt = [], []
        for i in range(bz):
            for output, trg_txt in zip(logist, trgs_):
                pred_txt = self.vocab.pred2s(output[i], oovs[i])
                preds_txt.append(pred_txt)
                trgs_txt.append(trg_txt[i])

        return {"loss": loss, "preds_txt": preds_txt, "trgs_txt": trgs_txt}

    def validation_step_end(self, batch_parts):
        loss = batch_parts["loss"].mean()
        self.log("valid/loss", loss, on_step=False, on_epoch=True)

        preds_txt, trgs_txt = [], []
        for pred, trg in zip(batch_parts["preds_txt"], batch_parts["trgs_txt"]):
            preds_txt.append(" ".join(pred))
            trgs_txt.append(trg)
        return {"preds_txt": preds_txt, "trgs_txt": trgs_txt}

    def validation_epoch_end(self, outputs) -> None:
        outputs = outputs[0]
        outputs = [
            {"pred": pred, "trg": trg}
            for pred, trg in zip(outputs["preds_txt"], outputs["trgs_txt"])
        ]

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

        self.log("valid/bleu_1", bleu_1)
        self.log("valid/bleu_4", bleu_4)
        self.log("valid/meteor", meteor)
        self.log("valid/rouge_l", rouge_l)

    def test_step(self, batch, batch_idx):
        _, logist = self.model.do_predict(**batch)
        # logist: [b, la]

        logist = [logist, logist] if self.use_2_answers else [logist]
        oovs = batch["oovs"]
        trgs_ = [batch["trg1_txt"], batch["trg2_txt"]] if len(logist) > 1 else [batch["trg1_txt"]]

        bz = batch["q_ids"].size(0)
        preds_txt, trgs_txt = [], []
        for i in range(bz):
            for output, trg_txt in zip(logist, trgs_):
                pred_txt = self.vocab.pred2s(output[i], oovs[i])
                preds_txt.append(pred_txt)
                trgs_txt.append(trg_txt[i])

        return {"preds_txt": preds_txt, "trgs_txt": trgs_txt}

    def test_step_end(self, batch_parts):
        loss = batch_parts["loss"].mean()
        self.log("valid/loss", loss, on_step=False, on_epoch=True)

        preds_txt, trgs_txt = [], []
        for pred, trg in zip(batch_parts["preds_txt"], batch_parts["trgs_txt"]):
            preds_txt.append(" ".join(pred))
            trgs_txt.append(trg)
        return {"preds_txt": preds_txt, "trgs_txt": trgs_txt}

    def test_epoch_end(self, outputs) -> None:
        outputs = outputs[0]
        outputs = [
            {"pred": pred, "trg": trg}
            for pred, trg in zip(outputs["preds_txt"], outputs["trgs_txt"])
        ]

        if not self.is_tuning:
            with open(self.path_test_pred, "a+") as pred_file:
                json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

        self.log("bleu_1", bleu_1)
        self.log("bleu_4", bleu_4)
        self.log("meteor", meteor)
        self.log("rouge_l", rouge_l)
