import json

import pytorch_lightning as plt
import torch
import torch.nn as torch_nn
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
from utils.model_utils import get_scores

from models.layers.chime import CHIME


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        lq,
        lc,
        la,
        d_hid,
        d_graph,
        d_bert,
        d_vocab,
        n_edges,
        n_propagations,
        lr,
        w_decay,
        warmup_rate,
        dropout,
        size_dataset_train,
        max_epochs,
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
            d_hid=d_hid,
            d_graph=d_graph,
            d_vocab=d_vocab,
            n_edges=n_edges,
            dropout=dropout,
            n_propagations=n_propagations,
            path_pretrained=path_pretrained,
        )

        #############################
        # Define things
        #############################
        self.criterion = torch_nn.CrossEntropyLoss(ignore_index=self.bert_tokenizer.pad_token_id)

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

    def training_step(self, batch, batch_idx):
        output_mle, trgs = self.model.do_train(
            batch["q_ids"], batch["c_ids"], batch["a1_ids"], batch["c_masks"]
        )
        # trgs: [b, la + 1]
        # output_mle: [b, d_vocab, la + 1]

        loss = self.criterion(output_mle, trgs)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        # output_mle: [b, la + 2, d_vocab]

        logist = [torch.argmax(output_mle, dim=1)]
        trgs_ = [batch["a1_ids"]]

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

    def training_step_end(self, batch_parts):
        size_pred = batch_parts["size_pred"][0]
        size_trg = batch_parts["size_trg"][0]

        preds, trgs = [], []
        for pred, trg in zip(batch_parts["pred"], batch_parts["trg"]):
            preds.extend(pred.view(-1, size_pred).cpu().detach())
            trgs.extend(trg.view(-1, size_trg).cpu().detach())

        return {"loss": batch_parts["loss"].mean(), "pred": preds, "trg": trgs}

    def training_epoch_end(self, outputs) -> None:
        outputs = outputs[0]

        ## Calculate mean loss
        self.log("train/loss_epoch", outputs["loss"])

        ## Calculate B-1, B-4, METEOR and ROUGE-L
        outputs = self.get_prediction(outputs)

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

        self.log("train/bleu_1", bleu_1)
        self.log("train/bleu_4", bleu_4)
        self.log("train/meteor", meteor)
        self.log("train/rouge_l", rouge_l)

    def test_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        pred = self.model.do_predict(batch["q_ids"], batch["c_ids"], batch["c_masks"], self.la)
        # logist: [b, la]

        logist = [pred, pred]
        trgs_ = [batch["a1_ids"], batch["a2_ids"]]

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
        size_pred = batch_parts["size_pred"][0]
        size_trg = batch_parts["size_trg"][0]

        preds, trgs = [], []
        for pred, trg in zip(batch_parts["pred"], batch_parts["trg"]):
            preds.extend(pred.view(-1, size_pred).cpu().detach())
            trgs.extend(trg.view(-1, size_trg).cpu().detach())

        return {"pred": preds, "trg": trgs}

    def validation_epoch_end(self, outputs) -> None:
        outputs = outputs[0]

        outputs = self.get_prediction(outputs)

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(outputs)

        # if self.trainer.is_global_zero:
        self.log("valid/bleu_1", bleu_1)
        self.log("valid/bleu_4", bleu_4)
        self.log("valid/meteor", meteor)
        self.log("valid/rouge_l", rouge_l)

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
