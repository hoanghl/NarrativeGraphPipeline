import json
from typing import Any, Optional

import pytorch_lightning as plt
import torch
import torch.nn as torch_nn
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from datamodules.narrative_datamodule import NarrativeDataModule
from models.layers.ans_infer_layer import BertDecoder
from models.layers.finegrain_layer import FineGrain
from models.layers.graph_layer import GraphBasedLayer
from utils.model_utils import get_scores, ipot


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        l_q,
        l_c,
        l_a,
        n_nodes,
        n_edges,
        n_gru_layers,
        d_hid,
        d_bert,
        d_vocab,
        d_graph,
        lr,
        w_decay,
        size_dataset_train,
        max_epochs,
        warmup_rate,
        path_pretrained,
        path_train_pred,
        path_valid_pred,
        datamodule: NarrativeDataModule = None,
        **kwargs
    ):

        super().__init__()

        self.batch_size = batch_size
        self.l_a = l_a
        self.d_vocab = d_vocab
        self.lr = lr
        self.w_decay = w_decay
        self.size_dataset_train = size_dataset_train
        self.max_epochs = max_epochs
        self.warmup_rate = warmup_rate

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_pretrained)
        self.datamodule = datamodule

        self.path_train_pred = path_train_pred
        self.path_valid_pred = path_valid_pred

        #############################
        # Define model
        #############################
        self.embd_layer = FineGrain(
            l_a=l_a,
            l_c=l_c,
            n_gru_layers=n_gru_layers,
            d_bert=d_bert,
            d_hid=d_hid,
            path_pretrained=path_pretrained,
        )
        self.reasoning = GraphBasedLayer(
            batch_size=batch_size,
            l_q=l_q,
            l_a=l_a,
            d_hid=d_hid,
            d_bert=d_bert,
            d_graph=d_graph,
            n_nodes=n_nodes,
            n_edges=n_edges,
        )
        self.ans_infer = BertDecoder(
            l_a=l_a,
            d_bert=d_bert,
            d_vocab=d_vocab,
            cls_tok_id=self.bert_tokenizer.cls_token_id,
            sep_tok_id=self.bert_tokenizer.sep_token_id,
            embd_layer=self.embd_layer,
        )

        ## Freeeze some parameters
        list_freeze_sets = [
            # self.embd_layer.bert_emb.parameters(),
            # self.ans_infer.decoder.parameters(),
        ]
        for params in list_freeze_sets:
            for param in params:
                param.requires_grad = False

        #############################
        # Define things
        #############################
        self.criterion = torch_nn.CrossEntropyLoss(ignore_index=self.bert_tokenizer.pad_token_id)

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def get_score_from_outputs(self, outputs):
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs:
            try:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = get_scores(**pair)
            except ValueError:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = 0, 0, 0, 0

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

            n_samples += 1

        return (
            bleu_1 / n_samples,
            bleu_4 / n_samples,
            meteor / n_samples,
            rouge_l / n_samples,
        )

    def get_prediction(self, output_mle, a1_ids, a2_ids):
        prediction = torch.argmax(output_mle, dim=1)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(prediction.squeeze(1), a1_ids, a2_ids)
        ]

        return prediction

    def get_loss(self, output_mle, output_ot, a_ids, a_masks, gamma=0.1):
        # output_mle: [b, d_vocab, l_a-1]
        # output_ot: [b, l_a-1, d_hid]
        # a_ids: [b, l_a]
        # a_masks: [b, l_a]

        # Calculate MLE loss
        loss_mle = self.criterion(output_mle, a_ids[:, 1:])

        # Calculate OT loss
        a = self.embd_layer.encode_ans(a_ids=a_ids, a_masks=a_masks, ot_loss=True)[:, 1:]
        # [b, l_a-1, d_hid]

        loss_ot = ipot(output_ot, a, max_iter=400)

        total_loss = loss_mle + gamma * loss_ot

        return total_loss

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
        # q_ids: [b, l_q]
        # q_masks  : [b, l_q]
        # c_ids: [b, n_c, l_c]
        # c_masks : [b, n_c, l_c]
        # a1_ids: [b, l_a]
        # a1_masks   : [b, l_a]

        ####################
        # Embed question, c and answer
        ####################
        q, c = self.embd_layer.encode_ques_para(
            q_ids=q_ids,
            c_ids=c_ids,
            q_masks=q_masks,
            c_masks=c_masks,
        )
        # q: [b, l_q, d_bert]
        # c: [b, n_c, d_bert]
        # a: [b, l_a, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(q, c)
        # [b, n_nodes, d_bert]

        ####################
        # Generate answer
        ####################
        return (
            self.ans_infer.do_train(
                Y=Y,
                a_ids=a1_ids,
                a_masks=a1_masks,
                cur_step=cur_step,
                max_step=max_step,
            )
            if not is_predict
            else self.ans_infer.do_predict(Y=Y)
        )

    def training_step(self, batch: Any, batch_idx: int):
        output_mle, output_ot = self(
            **batch,
            cur_step=batch_idx,
            max_step=self.datamodule.data_train.size_dataset // self.datamodule.batch_size,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]
        a1_masks = batch["a1_masks"]
        loss = self.get_loss(output_mle, output_ot, a1_ids, a1_masks)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return {
            "loss": loss,
            "pred": (
                output_mle.cpu().detach(),
                a1_ids.cpu().detach(),
                a2_ids.cpu().detach(),
            ),
        }

    def training_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(self.get_prediction(p[0], p[1], p[2]))

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = self.get_score_from_outputs(preds)

        self.log("train/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("train/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("train/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("train/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

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
        output_mle, output_ot = self(**batch, is_predict=True)
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]
        a1_masks = batch["a1_masks"]
        loss = self.get_loss(output_mle, output_ot, a1_ids, a1_masks)

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {
            "loss": loss,
            "pred": (
                output_mle.cpu().detach(),
                a1_ids.cpu().detach(),
                a2_ids.cpu().detach(),
            ),
        }

    def validation_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(self.get_prediction(p[0], p[1], p[2]))

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = self.get_score_from_outputs(preds)

        self.log("valid/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    # def on_validation_end(self) -> None:
    #     if self.current_epoch % self.switch_frequency == 0 and self.current_epoch != 0:
    #         self.datamodule.switch_answerability()

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int],
    ) -> Any:
        output_mle, _ = self(**batch, is_predict=True)
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]

        return {
            "pred": (
                output_mle.cpu().detach(),
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
