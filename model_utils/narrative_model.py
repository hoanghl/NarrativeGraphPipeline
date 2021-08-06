from typing import Any, Optional
import json

from transformers import AdamW, BertTokenizer
import pytorch_lightning as plt
import torch.nn as torch_nn
import torch


from data_utils.narrative_datamodule import NarrativeDataModule
from model_utils.layers.reasoning_layer import Reasoning
from model_utils.layers.bertbasedembd_layer import BertBasedEmbedding
from model_utils.layers.ans_infer_layer import Decoder
from utils.model_utils import ipot, get_scores

EPSILON = 10e-10


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        batch_size,
        l_q,
        l_c,
        l_a,
        n_heads,
        d_hid,
        d_bert,
        d_vocab,
        lr,
        w_decay,
        dropout,
        beam_size,
        n_gram_beam,
        path_pretrained,
        path_pred,
        path_train_pred,
        path_valid_pred,
        datamodule,
        **kwargs
    ):

        super().__init__()

        self.d_vocab = d_vocab
        self.lr = lr
        self.w_decay = w_decay
        self.beam_size = beam_size
        self.n_gram_beam = n_gram_beam
        self.l_a = l_a

        self.path_pred = path_pred
        self.path_train_pred = path_train_pred
        self.path_valid_pred = path_valid_pred

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_pretrained)
        self.datamodule: NarrativeDataModule = datamodule

        #############################
        # Define model
        #############################
        self.embd_layer = BertBasedEmbedding(
            d_bert=d_bert, d_hid=d_hid, path_pretrained=path_pretrained
        )
        self.reasoning = Reasoning(
            l_q=l_q,
            l_c=l_c,
            n_heads=n_heads,
            d_hid=d_hid,
            dropout=dropout,
            device=self.device,
        )
        self.ans_infer = Decoder(
            batch_size=batch_size,
            l_a=l_a,
            d_vocab=d_vocab,
            d_hid=d_hid,
            tokenizer=self.bert_tokenizer,
            embd_layer=self.embd_layer,
            beam_size=beam_size,
            device=self.device,
        )

        ## Freeeze some parameters
        list_freeze_sets = [
            self.embd_layer.bert_emb.parameters(),
            # self.ans_infer.decoder.parameters(),
        ]
        for params in list_freeze_sets:
            for param in params:
                param.requires_grad = False

        #############################
        # Define things
        #############################
        self.criterion = torch_nn.CrossEntropyLoss(
            ignore_index=self.bert_tokenizer.pad_token_id
        )

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    def get_loss(self, output_mle, output_ot, a_ids, a_masks, gamma=0.1):
        # output_mle: [b, d_vocab, l_a-1]
        # output_ot: [b, l_a-1, d_hid]
        # a_ids: [b, l_a-1]
        # a_masks: [b, l_a-1]

        # Calculate MLE loss
        loss_mle = self.criterion(output_mle, a_ids)

        # Calculate OT loss
        ans = self.embd_layer.encode_ans(input_ids=a_ids, input_masks=a_masks)
        # [b, l_a-1, d_hid]

        loss_ot = ipot(output_ot, ans, max_iter=500)

        total_loss = loss_mle + gamma * loss_ot

        return total_loss

    def get_prediction(self, output_mle, a1_ids, a2_ids):
        _, prediction = torch.topk(output_mle, 1, dim=1)

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
        # q_masks: [b, l_q]
        # c_ids: [b, n_c, l_c]
        # c_masks: [b, n_c, l_c]
        # a_ids: [b, l_a]
        # a_masks: [b, l_a]

        ####################
        # Embed question, c and answer
        ####################
        q, c = self.embd_layer.encode_ques_para(
            q_ids=q_ids,
            c_ids=c_ids,
            q_masks=q_masks,
            c_masks=c_masks,
        )
        # q : [b, l_q, d_hid]
        # c: [b, n_c, l_c, d_hid]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(q=q, c=c)
        # [b, 1, d_hid]

        ####################
        # Generate answer
        ####################

        return (
            self.ans_infer.do_predict(Y=Y)
            if is_predict
            else self.ans_infer.do_train(
                Y=Y,
                a_ids=a1_ids,
                a_masks=a1_masks,
                cur_step=cur_step,
                max_step=max_step,
            )
        )
        # pred: [b, d_vocab, l_a - 1]

    def training_step(self, batch: Any, batch_idx):
        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]
        a1_masks = batch["a1_masks"]

        output_mle, output_ot = self(
            **batch,
            cur_step=batch_idx,
            max_step=self.datamodule.data_train.size_dataset
            // self.datamodule.batch_size,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        loss = self.get_loss(output_mle, output_ot, a1_ids[:, 1:], a1_masks[:, 1:])

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

    def test_step(self, batch: Any, batch_idx):
        return 0

    def validation_step(self, batch: Any, batch_idx):
        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]
        a1_masks = batch["a1_masks"]

        output_mle, output_ot = self(
            **batch,
            is_predict=True,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        loss = self.get_loss(output_mle, output_ot, a1_ids[:, 1:], a1_masks[:, 1:])

        self.log(
            "valid/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            # sync_dist=True,
        )

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

    def on_validation_epoch_end(self) -> None:
        # TODO: Find the way to finish
        if self.current_epoch % 6 == 0 and self.current_epoch != 0:
            self.datamodule.switch_answerability()

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(), lr=self.lr, weight_decay=self.w_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=60, eta_min=0
            ),
        }

    def predict_step(
        self,
        batch: Any,
        batch_idx,
        dataloader_idx: Optional[int],
    ) -> Any:
        a1_ids = batch["a1_ids"]
        a2_ids = batch["a2_ids"]

        output_mle, _ = self(
            **batch,
            is_predict=True,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

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
