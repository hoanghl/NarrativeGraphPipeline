from typing import Any, Optional
from itertools import combinations
import json, re

from transformers import AdamW, BertTokenizer
import pytorch_lightning as plt
import torch.nn.functional as torch_F
import torch.nn as torch_nn
import torch
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

from src.datamodules.narrative_datamodule import NarrativeDataModule
from src.models.layers.reasoning_layer.memory_layer import MemoryBasedReasoning
from src.models.layers.bertbasedembd_layer import BertBasedLayer
from src.models.layers.ans_infer_layer import BertDecoder
from src.utils.generator import GeneratorOwn

EPSILON = 10e-10


class NarrativeModel(plt.LightningModule):
    def __init__(
        self,
        seq_len_ques: int = 42,
        seq_len_para: int = 162,
        seq_len_ans: int = 42,
        max_len_ans: int = 12,
        n_paras: int = 30,
        n_layers_gru: int = 4,
        n_layers_trans: int = 3,
        n_heads_trans: int = 4,
        d_hid: int = 64,
        d_bert: int = 768,
        d_vocab: int = 30522,
        lr: float = 1e-5,
        w_decay: float = 1e-2,
        beam_size: int = 20,
        n_gram_beam: int = 5,
        path_bert: str = None,
        path_pred: str = None,
        path_train_pred: str = None,
        datamodule: NarrativeDataModule = None,
    ):

        super().__init__()

        self.d_vocab = d_vocab
        self.lr = lr
        self.w_decay = w_decay
        self.beam_size = beam_size
        self.n_gram_beam = n_gram_beam
        self.max_len_ans = max_len_ans

        self.bert_tokenizer = BertTokenizer.from_pretrained(path_bert)
        self.datamodule = datamodule

        self.path_pred = path_pred
        self.path_train_pred = path_train_pred

        #############################
        # Define model
        #############################
        self.embd_layer = BertBasedLayer(d_bert, path_bert)
        self.reasoning = MemoryBasedReasoning(
            seq_len_ques,
            seq_len_para,
            seq_len_ans,
            n_paras,
            n_layers_gru,
            n_heads_trans,
            n_layers_trans,
            d_hid,
            d_bert,
            self.device,
        )
        self.ans_infer = BertDecoder(seq_len_ans, d_bert, d_vocab)

        ## Freeeze some parameters
        list_freeze_sets = [
            self.embd_layer.bert_emb.parameters(),
            # self.ans_infer.decoder.parameters(),
        ]
        for params in list_freeze_sets:
            for param in params:
                param.requires_grad = True

        #############################
        # Define things
        #############################
        self.criterion = torch_nn.CrossEntropyLoss(
            ignore_index=self.bert_tokenizer.pad_token_id
        )

    ####################################################################
    # FOR TRAINING PURPOSE
    ####################################################################

    # Some utils
    def calc_loss(self, pred, ans):
        # pred: [b, seq_len_ans, d_vocab]
        # ans: [b, seq_len_ans]

        d_vocab = pred.shape[2]

        pred_flat = pred[:, :-1, :].reshape(-1, d_vocab)
        ans_flat = ans[:, 1:].reshape(-1)

        loss = self.criterion(pred_flat, ans_flat)

        return loss

    def process_sent(self, sent: str):
        return re.sub(r"(\[PAD\]|\[CLS\]|\[SEP\]|\[UNK\]|\[MASK\])", "", sent).strip()

    def get_scores(self, ref: list, pred: str):
        """Calculate metrics BLEU-1, BLEU4, METEOR and ROUGE_L.

        ref = [
            "the transcript is a written version of each day",
            "version of each day"
        ]
        pred= "a written version of"

        Args:
            ref (list): list of reference strings
            pred (str): string generated by model

        Returns:
            tuple: tuple of 4 scores



        """
        pred = self.process_sent(pred)
        ref = list(map(self.process_sent, ref))

        # Calculate BLEU score
        ref_ = [x.split() for x in ref]
        pred_ = pred.split()

        bleu_1 = sentence_bleu(ref_, pred_, weights=(1, 0, 0, 0))
        bleu_4 = sentence_bleu(ref_, pred_, weights=(0.25, 0.25, 0.25, 0.25))

        # Calculate METEOR
        meteor = meteor_score(ref, pred)

        # Calculate ROUGE-L
        scores = np.array(
            [Rouge().get_scores(ref_, pred, avg=True)["rouge-l"]["f"] for ref_ in ref]
        )
        rouge_l = np.mean(scores)

        return (
            bleu_1 if bleu_1 > EPSILON else 0,
            bleu_4 if bleu_4 > EPSILON else 0,
            meteor if meteor > EPSILON else 0,
            rouge_l if rouge_l > EPSILON else 0,
        )

    def calc_diff_ave(self, Y: torch.Tensor):
        b = Y.shape[0]

        diff_ave, n = 0, 0
        for (i, j) in combinations(list(range(b)), 2):
            diff_ave += torch.linalg.norm(Y[i] - Y[j]).item()
            n += 1

        return diff_ave / n

    def model(self, ques, ques_mask, ans, ans_mask, paras, paras_mask):
        # ques       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # paras      : [b, n_paras, seq_len_para]
        # paras_mask : [b, n_paras, seq_len_para]
        # ans        : [b, seq_len_ans]
        # ans_mask   : [b, seq_len_ans]

        ####################
        # Embed question, paras and answer
        ####################
        ques, paras = self.embd_layer.encode_ques_para(
            ques, paras, ques_mask, paras_mask
        )
        ans = self.embd_layer.encode_ans(ans, ans_mask)
        # ques : [b, seq_len_ques, d_bert]
        # paras: [b, n_paras, seq_len_para, d_bert]
        # ans  : [b, seq_len_ans, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, paras, paras_mask)
        # [b, seq_len_ans, d_bert]

        ####################
        # Generate answer
        ####################
        pred = self.ans_infer(Y, ans, ans_mask)
        # pred: [b, seq_len_ans, d_vocab]

        return pred, self.calc_diff_ave(Y)

    def training_step(self, batch: Any, batch_idx: int):
        ques = batch["ques"]
        ques_mask = batch["ques_mask"]
        ans1 = batch["ans1"]
        ans2 = batch["ans2"]
        ans1_mask = batch["ans1_mask"]
        paras = batch["paras"]
        paras_mask = batch["paras_mask"]

        pred, diff_ave = self.model(ques, ques_mask, ans1, ans1_mask, paras, paras_mask)

        loss = self.calc_loss(pred, ans1)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/diff_ave", diff_ave, on_step=True, on_epoch=True, prog_bar=True)

        _, prediction = torch.topk(torch_F.log_softmax(pred, dim=2), 1, dim=2)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ans1": " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                "ans2": " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
            }
            for pred_, ans1_, ans2_ in zip(prediction, ans1, ans2)
        ]

        return {"loss": loss, "pred": prediction}

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with open(self.path_train_pred, "a+") as pred_file:
            json.dump(outputs["pred"], pred_file, indent=2, ensure_ascii=False)

    def test_step(self, batch: Any, batch_idx: int):
        ques = batch["ques"]
        ques_mask = batch["ques_mask"]
        ans1 = batch["ans1"]
        ans1_mask = batch["ans1_mask"]
        paras = batch["paras"]
        paras_mask = batch["paras_mask"]

        pred = self.model(ques, ques_mask, ans1, ans1_mask, paras, paras_mask)

        loss = self.calc_loss(pred, ans1)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        ques = batch["ques"]
        ques_mask = batch["ques_mask"]
        ans1 = batch["ans1"]
        ans2 = batch["ans2"]
        ans1_mask = batch["ans1_mask"]
        paras = batch["paras"]
        paras_mask = batch["paras_mask"]

        pred, _ = self.model(ques, ques_mask, ans1, ans1_mask, paras, paras_mask)
        # pred: [b, seq_len_ans, d_vocab]

        loss = self.calc_loss(pred, ans1)

        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        _, prediction = torch.topk(torch_F.log_softmax(pred, dim=2), 1, dim=2)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ref": [
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                    " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
                ],
            }
            for pred_, ans1_, ans2_ in zip(prediction, ans1, ans2)
        ]

        return {"loss": loss, "pred": prediction}

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        n_samples = 0
        bleu_1, bleu_4, meteor, rouge_l = 0, 0, 0, 0
        for pair in outputs["pred"]:
            bleu_1_, bleu_4_, meteor_, rouge_l_ = self.get_scores(**pair)

            bleu_1 += bleu_1_
            bleu_4 += bleu_4_
            meteor += meteor_
            rouge_l += rouge_l_

            n_samples += 1

        self.log("valid/bleu_1", bleu_1 / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4 / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor / n_samples, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l / n_samples, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(), lr=self.lr, weight_decay=self.w_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=15, eta_min=0
            ),
        }

    #########################################
    # FOR PREDICTION PURPOSE
    #########################################
    def forward(self, ques, ques_mask, paras, paras_mask):
        # ques       : [b, seq_len_ques]
        # ques_mask  : [b, seq_len_ques]
        # paras      : [b, n_paras, seq_len_para]
        # paras_mask : [b, n_paras, seq_len_para]

        b = ques.shape[0]

        ####################
        # Embed question, paras and answer
        ####################
        ques, paras = self.embd_layer.encode_ques_para(
            ques, paras, ques_mask, paras_mask
        )
        # ques : [b, seq_len_ques, d_bert]
        # paras: [b, n_paras, d_bert]

        ####################
        # Do reasoning
        ####################
        Y = self.reasoning(ques, paras)
        # [b, seq_len_ans, d_bert]

        ####################
        # Generate answer
        ####################
        outputs = []

        generator = GeneratorOwn(
            beam_size=self.beam_size,
            init_tok=self.bert_tokenizer.cls_token_id,
            stop_tok=self.bert_tokenizer.sep_token_id,
            max_len=self.max_len_ans,
            model=self.generate,
            no_repeat_ngram_size=self.n_gram_beam,
            topk_strategy="select_mix_beam",
        )

        for b_ in range(b):
            indices = generator.search(Y[b_, :, :])

            outputs.append(indices)

        outputs = torch.tensor(outputs, device=self.device, dtype=torch.long)

        return outputs

    def generate(self, decoder_input_ids, encoder_outputs):
        # decoder_input_ids: [seq_len<=200]
        # encoder_outputs  : [seq_len_ans, d_bert]

        decoder_input_ids = (
            torch.LongTensor(decoder_input_ids)
            .type_as(encoder_outputs)
            .long()
            .unsqueeze(0)
        )

        decoder_input_mask = torch.ones(decoder_input_ids.shape, device=self.device)
        decoder_input_embd = self.embd_layer.encode_ans(
            decoder_input_ids, decoder_input_mask
        )
        # [1, seq=*, d_bert]

        encoder_outputs = encoder_outputs.unsqueeze(0)

        output = self.ans_infer(encoder_outputs, decoder_input_embd, decoder_input_mask)
        # [1, seq=*, d_vocab]

        output = output.squeeze(0)
        # [seq=*, d_vocab]

        return output

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int],
    ) -> Any:
        ques = batch["ques"]
        ques_mask = batch["ques_mask"]
        paras = batch["paras"]
        paras_mask = batch["paras_mask"]
        ans1 = batch["ans1"]
        ans2 = batch["ans2"]

        pred = self(ques, ques_mask, paras, paras_mask)

        prediction = [
            {
                "pred": " ".join(self.bert_tokenizer.convert_ids_to_tokens(pred_)),
                "ans1": " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans1_)),
                "ans2": " ".join(self.bert_tokenizer.convert_ids_to_tokens(ans2_)),
            }
            for pred_, ans1_, ans2_ in zip(pred, ans1, ans2)
        ]

        return prediction

    def on_predict_batch_end(
        self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with open(self.path_pred, "a+") as pred_file:
            json.dump(outputs, pred_file, indent=2, ensure_ascii=False)
