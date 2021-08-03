from typing import Any, Optional
import json


from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
import pytorch_lightning as plt
import torch.nn.functional as torch_F
import torch.nn as torch_nn
import torch

from datamodules.narrative_datamodule import NarrativeDataModule
from models.layers.reasoning_layer.memorygraph_layer import GraphBasedMemoryLayer
from models.layers.finegrain_layer import FineGrain
from models.layers.ans_infer_layer import BertDecoder
from utils.model_utils import ipot, get_scores


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
        n_epochs,
        size_dataset_train,
        warmup_rate,
        lr,
        switch_frequency,
        beam_size,
        n_gram_beam,
        path_pretrained,
        path_train_pred,
        path_valid_pred,
        datamodule: NarrativeDataModule = None,
        **kwargs
    ):

        super().__init__()

        self.d_vocab = d_vocab
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.size_dataset_train = size_dataset_train
        self.warmup_rate = warmup_rate
        self.lr = lr
        self.beam_size = beam_size
        self.n_gram_beam = n_gram_beam
        self.l_a = l_a
        self.switch_frequency = switch_frequency

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
        self.reasoning = GraphBasedMemoryLayer(
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
            sep_tok_id=self.bert_tokenizer.sep_token,
            beam_size=beam_size,
            n_gram_beam=n_gram_beam,
            embd_layer=self.embd_layer,
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
        _, prediction = torch.topk(torch_F.log_softmax(output_mle, dim=1), 1, dim=1)

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

    def get_loss(self, output_mle, output_ot, a_ids, a_masks, gamma=0.05):
        # output_mle: [b, d_vocab, l_a-1]
        # output_ot: [b, l_a-1, d_hid]
        # a_ids: [b, l_a]
        # a_masks: [b, l_a]

        # Calculate MLE loss
        loss_mle = self.criterion(output_mle, a_ids[:, 1:])

        # Calculate OT loss
        ans = self.embd_layer.encode_ans(ans_ids=a_ids, ans_mask=a_masks, ot_loss=True)[
            :, 1:
        ]
        # [b, l_a-1, d_hid]

        loss_ot = ipot(output_ot, ans)

        total_loss = loss_mle + gamma * loss_ot

        return total_loss

    def forward(
        self,
        ques_ids,
        ques_mask,
        context_ids,
        context_mask,
        ans1_ids=None,
        ans1_mask=None,
        cur_step=0,
        max_step=0,
        is_predict=False,
        **kwargs
    ):
        # ques       : [b, l_q]
        # ques_mask  : [b, l_q]
        # context_ids  : [b, n_paras, l_c]
        # context_mask : [b, n_paras, l_c]
        # ans1_ids: [b, l_a]
        # ans1_mask   : [b, l_a]

        ####################
        # Embed question, context and answer
        ####################
        ques, context = self.embd_layer.encode_ques_para(
            ques_ids=ques_ids,
            context_ids=context_ids,
            ques_mask=ques_mask,
            context_mask=context_mask,
        )
        # ques : [b, l_q, d_bert]
        # context: [b, n_paras, d_bert]
        # ans  : [b, l_a, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, context)
        # [b, l_a, d_bert]

        ####################
        # Generate answer
        ####################
        return (
            self.ans_infer.do_train(
                Y=Y,
                ans_ids=ans1_ids,
                ans_mask=ans1_mask,
                cur_step=cur_step,
                max_step=max_step,
            )
            if not is_predict
            else self.ans_infer.do_predict(Y)
        )

    def training_step(self, batch: Any, batch_idx: int):
        output_mle, output_ot = self(
            **batch,
            cur_step=batch_idx,
            max_step=self.datamodule.data_train.size_dataset
            // self.datamodule.batch_size,
        )
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        ans1_mask = batch["ans1_mask"]
        loss = self.get_loss(output_mle, output_ot, ans1_ids, ans1_mask)

        prediction = self.get_prediction(output_mle, ans1_ids, ans2_ids)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/sampling",
            self.ans_infer.t,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        return {"loss": loss, "pred": prediction}

    def training_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(p)

        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = self.get_score_from_outputs(preds)

        self.log("train/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("train/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("train/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("train/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params1, params2 = [], []
        for layer in [self.embd_layer, self.reasoning, self.ans_infer]:
            for n, p in layer.named_parameters():
                if not any(nd in n for nd in no_decay):
                    params1.append(p)
                else:
                    params2.append(p)
        optimizer_grouped_parameters = [
            {
                "params": params1,
                "weight_decay": 0.95,
            },
            {
                "params": params2,
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.lr)

        n_training_steps = self.size_dataset_train // self.batch_size * self.n_epochs
        return {
            "optimizer": optimizer,
            "lr_scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(n_training_steps * self.warmup_rate),
                num_training_steps=n_training_steps,
            ),
        }

    #########################################
    # FOR PREDICTION PURPOSE
    #########################################

    def validation_step(self, batch: Any, batch_idx: int):
        output_mle, output_ot = self(**batch, is_predict=True)
        # output_ot: [b, l_a - 1, d_hid]
        # output_mle: [b, d_vocab, l_a - 1]

        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        ans1_mask = batch["ans1_mask"]
        loss = self.get_loss(output_mle, output_ot, ans1_ids, ans1_mask)

        prediction = self.get_prediction(output_mle, ans1_ids, ans2_ids)

        self.log("valid/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": loss, "pred": prediction}

    def validation_epoch_end(self, outputs) -> None:
        preds = []
        for p in [out["pred"] for out in outputs]:
            preds.extend(p)

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = self.get_score_from_outputs(preds)

        self.log("valid/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l, on_epoch=True, prog_bar=False)

    def on_validation_end(self) -> None:
        if self.current_epoch % self.switch_frequency == 0 and self.current_epoch != 0:
            self.datamodule.switch_answerability()

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int],
    ) -> Any:
        ques_ids = batch["ques_ids"]
        ques_mask = batch["ques_mask"]
        ans1_ids = batch["ans1_ids"]
        ans2_ids = batch["ans2_ids"]
        context_ids = batch["context_ids"]
        context_mask = batch["context_mask"]

        prediction = self(
            ques_ids=ques_ids,
            ques_mask=ques_mask,
            context_ids=context_ids,
            context_mask=context_mask,
        )

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(prediction.squeeze(1), ans1_ids, ans2_ids)
        ]

        return prediction

    def on_predict_batch_end(
        self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        #######################
        # Calculate metrics
        #######################
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs:
            try:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_scores(**pair)
            except ValueError:
                bleu_1_, bleu_4_, meteor_, rouge_l_ = 0, 0, 0, 0

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

            n_samples += 1

        #######################
        # Log prediction and metrics
        #######################
        with open(self.path_pred, "a+") as pred_file:
            json.dump(
                {
                    "metrics": {
                        "bleu_1": bleu_1 / n_samples,
                        "bleu_4": bleu_4 / n_samples,
                        "meteor": meteor / n_samples,
                        "rouge_l": rouge_l / n_samples,
                    },
                    "predictions": outputs,
                },
                pred_file,
                indent=2,
                ensure_ascii=False,
            )
