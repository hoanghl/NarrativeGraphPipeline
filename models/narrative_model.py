import json

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
        lc,
        la,
        nc,
        d_bert,
        d_hid,
        d_vocab,
        num_layers_lstm,
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
        self.model = Backbone(
            batch_size=batch_size,
            la=la,
            lc=lc,
            nc=nc,
            d_bert=d_bert,
            d_hid=d_hid,
            d_vocab=d_vocab,
            num_layers_lstm=num_layers_lstm,
            block=block,
            dropout=dropout,
            path_pretrained=path_pretrained,
            criterion=torch_nn.CrossEntropyLoss(ignore_index=self.bert_tokenizer.pad_token_id),
            device=self.device,
        )

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################
    def get_prediction(self, pairs):
        pairs = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pair["pred"])),
                "trg": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pair["trg"])),
            }
            for pair in pairs
        ]

        return pairs

    def training_step(self, batch, batch_idx: int):
        loss, logist = self.model.do_train(**batch)
        # logist: list of [b, la, d_vocab]

        if self.trainer.is_global_zero:
            self.log(
                "train/loss",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                rank_zero_only=True,
            )
        # logist: list of [b, la, d_vocab]

        logist = [torch.argmax(logist_, dim=1) for logist_ in logist]
        trgs = [batch["a1_ids"], batch["a2_ids"]] if len(logist) > 1 else [batch["a1_ids"]]

        bz = batch["q_ids"].size(0)
        preds = []
        for i in range(bz):
            for output, trg in zip(logist, trgs):
                preds.append(
                    {
                        "pred": output[i].cpu().detach().numpy(),
                        "trg": trg[i].cpu().detach().numpy(),
                    }
                )

        return {"loss": loss, "prediction": preds}

    def training_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)

        if self.trainer.is_global_zero:
            ## Calculate mean loss
            loss = torch.mean(torch.cat([output["loss"] for output in outputs]))
            self.log("train/loss", loss, rank_zero_only=True)

            ## Calculate B-1, B-4, METEOR and ROUGE-L
            output_ = []
            for output in outputs:
                output_.extend(output["prediction"])
            outputs = []
            for output in output_:
                if len(output["pred"].size()) == 2:
                    for b in range(output["pred"].size(0)):
                        outputs.append({"pred": output["pred"][b], "trg": output["trg"][b]})
                else:
                    outputs.append(output)

            outputs = self.get_prediction(outputs)

            with open(self.path_train_pred, "a+") as pred_file:
                json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

            bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

            self.log("train/bleu_1", bleu_1, rank_zero_only=True)
            self.log("train/bleu_4", bleu_4, rank_zero_only=True)
            self.log("train/meteor", meteor, rank_zero_only=True)
            self.log("train/rouge_l", rouge_l, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        return 0

    def validation_step(self, batch, batch_idx):
        loss, logist = self.model.do_predict(**batch)
        # logist: [b, la]

        logist = [logist, logist]
        trgs = [batch["a1_ids"], batch["a2_ids"]]

        bz = batch["q_ids"].size(0)
        preds = []
        for i in range(bz):
            for output, trg in zip(logist, trgs):
                preds.append(
                    {
                        "pred": output[i].cpu().detach().numpy(),
                        "trg": trg[i].cpu().detach().numpy(),
                    }
                )

        return {"loss": loss, "prediction": preds}

    def validation_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)

        if self.trainer.is_global_zero:
            ## Calculate mean loss
            loss = torch.mean(torch.cat([output["loss"] for output in outputs]))
            self.log("valid/loss", loss, rank_zero_only=True)

            ## Calculate B-1, B-4, METEOR and ROUGE-L
            output_ = []
            for output in outputs:
                output_.extend(output["prediction"])
            outputs = []
            for output in output_:
                if len(output["pred"].size()) == 2:
                    for b in range(output["pred"].size(0)):
                        outputs.append({"pred": output["pred"][b], "trg": output["trg"][b]})
                else:
                    outputs.append(output)

            outputs = self.get_prediction(outputs)

            with open(self.path_valid_pred, "a+") as pred_file:
                json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

            bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

            # if self.trainer.is_global_zero:
            self.log("valid/bleu_1", bleu_1, rank_zero_only=True)
            self.log("valid/bleu_4", bleu_4, rank_zero_only=True)
            self.log("valid/meteor", meteor, rank_zero_only=True)
            self.log("valid/rouge_l", rouge_l, rank_zero_only=True)

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
