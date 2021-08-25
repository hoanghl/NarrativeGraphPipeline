import json
from typing import Any, Optional

import pytorch_lightning as plt
import torch
import torch.nn as torch_nn
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from models.layers.ans_infer_layer import Decoder
from models.layers.finegrain_layer import FineGrain
from models.layers.graph_layer import GraphBasedReasoningLayer
from utils.model_utils import get_scores, ipot


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        la,
        n_nodes,
        n_edges,
        n_gru_layers,
        d_hid,
        d_bert,
        d_vocab,
        d_graph,
        lr,
        w_decay,
        dropout,
        size_dataset_train,
        max_epochs,
        warmup_rate,
        path_pretrained,
        path_train_pred,
        path_valid_pred,
        **kwargs
    ):

        super().__init__()

        self.batch_size = batch_size
        self.d_vocab = d_vocab
        self.lr = lr
        self.w_decay = w_decay
        self.size_dataset_train = size_dataset_train
        self.max_epochs = max_epochs
        self.warmup_rate = warmup_rate

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_pretrained)

        self.path_train_pred = path_train_pred
        self.path_valid_pred = path_valid_pred

        #############################
        # Define model
        #############################
        self.embd_layer = FineGrain(
            la=la,
            n_gru_layers=n_gru_layers,
            d_bert=d_bert,
            d_hid=d_hid,
            path_pretrained=path_pretrained,
        )
        # self.reasoning = GraphBasedReasoningLayer(
        #     batch_size=batch_size,
        #     d_hid=d_hid,
        #     d_bert=d_bert,
        #     d_graph=d_graph,
        #     n_nodes=n_nodes,
        #     n_edges=n_edges,
        #     dropout=dropout,
        # )
        self.ans_infer = Decoder(
            la=la,
            d_bert=d_bert,
            d_vocab=d_vocab,
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
        # q_masks  : [b, lq]
        # c_ids: [b, nc, lc]
        # c_masks : [b, nc, lc]
        # a1_ids: [b, la]
        # a1_masks   : [b, la]

        ####################
        # Embed question, c and answer
        ####################
        q, c = self.embd_layer.encode_ques_para(
            q_ids=q_ids,
            c_ids=c_ids,
            q_masks=q_masks,
            c_masks=c_masks,
        )
        # q: [b, lq, d_bert]
        # c: [b, nc, lc, d_bert]

        ####################
        # Do reasoning
        ####################
        # Y = self.reasoning(c)
        # [b, nc*lc, d_bert]

        ####################
        # Generate answer
        ####################
        return (
            self.ans_infer.do_train(
                Y=c,
                a_ids=a1_ids,
                a_masks=a1_masks,
                cur_step=cur_step,
                max_step=max_step,
            )
            if not is_predict
            else self.ans_infer.do_predict(
                Y=c,
                a_ids=a1_ids,
                a_masks=a1_masks,
            )
        )

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

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay, params_nodecay = [], []
        for model in [self.embd_layer, self.ans_infer]:
            for n, p in model.named_parameters():
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

    #########################################
    # FOR PREDICTION PURPOSE
    #########################################
    def validation_step(self, batch: Any, batch_idx: int):
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
