import json
import os

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup
from utils.model_utils import get_scores

from models.layers.chime import CHIME


class CHIMELitModel(LightningModule):
    def __init__(
        self,
        batch_size,
        l_a,
        max_epochs,
        path_pretrained,
        path_valid_pred,
        size_dataset_train,
        warmup_rate,
        w_decay,
        lr,
    ) -> None:
        super().__init__()

        # Define class attributes
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.size_dataset_train = size_dataset_train
        self.warmup_rate = warmup_rate
        self.single_answer_length = l_a
        self.path_valid_pred = path_valid_pred
        self.lr = lr
        self.w_decay = w_decay

        self.tokenizer = AutoTokenizer.from_pretrained(path_pretrained)
        self.beam_size = 1

        # Define model
        self.model = CHIME(path_pretrained)

        os.makedirs(path_valid_pred, exist_ok=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.w_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
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

    def training_step(self, batch, batch_idx):
        q, r, a = batch["q"], batch["c"], batch["a"]

        # Pass data through model
        lm_losses, _ = self.model((q, r, a))

        # Log
        self.log("train/loss", lm_losses, on_step=True, on_epoch=True, prog_bar=False)

        return lm_losses

    def validation_step(self, batch, batch_idx):
        que, rev, ans = batch["q"], batch["c"], batch["a"]
        predictions = []
        outputs = []
        for k in range(que.size(0)):
            top_candidates = []
            seen_words = {}

            # Prepare data
            q = que[k : k + 1, :]
            r = rev[k : k + 1, :, :]
            output = torch.tensor([[[]]], dtype=torch.long, device=self.device)

            # generate predictions
            _, output_ts = self.model((q, r, output))
            output_tbv, output_tbi = torch.topk(
                F.log_softmax(output_ts, dim=1), self.beam_size, dim=1
            )
            for p in range(self.beam_size):
                top_candidates.append(
                    ([output_tbi[:, p]], output_tbv[:, p])
                )  # save generation & beam scores
                seen_words[p] = {int(output_tbi[:, p].item()): 1}  # limit repeated words
            for t in range(self.single_answer_length - 1):
                candidates = {}
                for i in range(self.beam_size):
                    current_tokens = top_candidates[i][0]
                    if int(current_tokens[-1].item()) == self.tokenizer.sep_token_id:
                        pass
                    else:
                        output_c = torch.tensor(
                            current_tokens, dtype=torch.float, device=self.device
                        ).view(1, 1, -1)
                        _, output_cs = self.model((q, r, output_c))
                        output_cbv, output_cbi = torch.topk(
                            F.log_softmax(output_cs, dim=1), self.beam_size, dim=1
                        )
                        for j in range(self.beam_size):
                            current_p = output_cbi[:, j]
                            current_p_i = int(current_p.item())
                            current_p_c = seen_words[i].get(current_p_i, 0)
                            if current_p != current_tokens[-1] and current_p_c <= 1:
                                seen_words[i][current_p_i] = current_p_c + 1
                                beam_score = (
                                    top_candidates[i][1] * len(current_tokens) + output_cbv[:, j]
                                ) / ((len(current_tokens) + 1) ** 0.7)
                                candidates[(i, j)] = (current_tokens + [current_p], beam_score)
                            else:
                                pass
                if len(candidates) == 0:
                    break
                top_beams = sorted(candidates.items(), key=lambda x: -x[1][1])[
                    : self.beam_size
                ]  # beam_score sorting
                for top_beam_i, top_beam in enumerate(top_beams):
                    top_candidates[top_beam_i] = top_beam[1]
            pred = top_candidates[0][0]

            predictions.append(
                {
                    "ref": [" ".join(self.tokenizer.convert_ids_to_tokens(ans[k][0]))],
                    "pred": " ".join(self.tokenizer.convert_ids_to_tokens(pred)),
                }
            )

        return predictions

    def validation_epoch_end(self, outputs) -> None:
        preds = []
        for p in outputs:
            preds.extend(p)

        with open(self.path_valid_pred, "a+") as pred_file:
            json.dump(preds, pred_file, indent=2, ensure_ascii=False)

        bleu_1, bleu_4, meteor, rouge_l = get_scores(preds)
        self.log("valid/bleu_1", bleu_1, on_epoch=True, prog_bar=False)
        self.log("valid/bleu_4", bleu_4, on_epoch=True, prog_bar=False)
        self.log("valid/meteor", meteor, on_epoch=True, prog_bar=False)
        self.log("valid/rouge_l", rouge_l, on_epoch=True, prog_bar=False)
