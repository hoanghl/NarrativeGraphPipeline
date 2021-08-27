import json

import pytorch_lightning as plt
import torch
import torch.nn as torch_nn
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
from utils.model_utils import get_scores

from models.chime import CHIME


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        lq,
        lc,
        la,
        d_bert,
        d_vocab,
        lr,
        w_decay,
        max_epochs,
        warmup_rate,
        size_dataset_train,
        path_pretrained,
        path_valid_pred,
        path_train_pred,
    ):
        super().__init__()

        self.la = la
        self.lr = lr
        self.w_decay = w_decay
        self.warmup_rate = warmup_rate
        self.path_valid_pred = path_valid_pred
        self.path_train_pred = path_train_pred
        self.n_training_steps = size_dataset_train // batch_size * max_epochs
        self.bert_tokenizer = BertTokenizer.from_pretrained(path_pretrained)

        #############################
        # Define model
        #############################
        self.model = CHIME(
            lq=lq,
            lc=lc,
            d_bert=d_bert,
            d_vocab=d_vocab,
            path_pretrained=path_pretrained,
            criterion=torch_nn.CrossEntropyLoss(ignore_index=self.bert_tokenizer.pad_token_id),
        )

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

    def training_step(self, batch, batch_idx: int):
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

    def test_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
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
