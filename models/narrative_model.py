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

    def training_step(self, batch: Any, batch_idx: int):
        loss, logist = self.model.do_train(
            q=batch["q_ids"],
            c=batch["c_ids"],
            a1=batch["a1_ids"],
            a2=batch["a2_ids"],
            a1_mask=batch["a1_masks"],
            a2_mask=batch["a2_masks"],
            use_2_ans=True,
        )
        # logist: list of [b, la, d_vocab]

        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=False)

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
            self.log("train/loss_epoch", loss, rank_zero_only=True)

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

    def test_step(self, batch: Any, batch_idx):
        return 0

    def validation_step(self, batch: Any, batch_idx):
        logist = self.model.do_predict(batch["q_ids"], batch["c_ids"], self.la)
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

        return {"prediction": preds}

    def validation_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)

        # if self.trainer.is_global_zero:
        ## Calculate mean loss
        # loss = torch.mean(torch.cat([output["loss"] for output in outputs]))
        # self.log("valid/loss", loss, rank_zero_only=True)

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
        self.log("valid/bleu_1", bleu_1, sync_dist=True)
        self.log("valid/bleu_4", bleu_4, sync_dist=True)
        self.log("valid/meteor", meteor, sync_dist=True)
        self.log("valid/rouge_l", rouge_l, sync_dist=True)

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
