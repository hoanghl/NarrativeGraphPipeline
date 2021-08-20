import re

import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from transformers.generation_utils import *


def ipot(a1, a2, beta=2, max_iter=100, L=1):
    """Calculate loss based on OT."""

    b, l_a, d_hid = a1.size()
    n = b * l_a

    # a1: [b, l_a, d_hid]
    # a2: [b, l_a, d_hid]

    a1, a2 = a1.reshape(-1, d_hid), a2.reshape(-1, d_hid)
    # [n, d_hid]

    # Calculate matrix C
    a1_norm = a1 / a1.norm(dim=1)[:, None]
    a2_norm = a2 / a2.norm(dim=1)[:, None]
    C = a1_norm @ a2_norm.transpose(0, 1)
    # [n, n]

    sigma = torch.ones((n, 1), device=a1.device) / n

    T = torch.ones((n, n), device=a1.device) / n ** 2
    # [n, n]
    A = torch.exp(-(C / beta))
    # [n, n]

    for _ in range(max_iter):
        Q = A * T
        # [n, n]

        for _ in range(L):
            d = 1 / n / (Q @ sigma)
            sigma = 1 / n / (Q.T @ d)

        d1 = torch.diag(d.squeeze(1))
        d2 = torch.diag(sigma.squeeze(1))
        T = d1 * Q * d2

    loss = torch.sum(T * C)

    return loss


def process_sent(sent):
    return re.sub(r"(\[PAD\]|\[CLS\]|\[SEP\]|\[UNK\]|\[MASK\])", "", sent).strip()


def get_scores(ref: list, pred, eps=10e-8):
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

    pred = process_sent(pred)
    ref = list(map(process_sent, ref))

    if pred == "":
        return 0, 0, 0, 0

    # Calculate BLEU score
    ref_ = [x.split() for x in ref]
    pred_ = pred.split()

    bleu_1 = sentence_bleu(ref_, pred_, weights=(1, 0, 0, 0))
    bleu_4 = sentence_bleu(ref_, pred_, weights=(0.25, 0.25, 0.25, 0.25))

    # Calculate METEOR
    meteor = meteor_score(ref, pred)

    # Calculate ROUGE-L
    scores = np.array(
        [
            Rouge().get_scores(ref_, pred, avg=True)["rouge-l"]["f"] if ref != "" else 0
            for ref_ in ref
        ]
    )
    # scores = np.array(
    #     [Rouge().get_scores(ref_, pred, avg=True)["rouge-l"]["f"] for ref_ in ref]
    # )
    rouge_l = np.mean(scores)

    return (
        bleu_1 if bleu_1 > eps else 0,
        bleu_4 if bleu_4 > eps else 0,
        meteor if meteor > eps else 0,
        rouge_l if rouge_l > eps else 0,
    )


class GeneratorHugging(GenerationMixin):
    def __init__(
        self,
        batch_size,
        min_length,
        max_length,
        num_beams,
        temperature,
        no_repeat_ngram_size,
        device,
        model: Any = None,
        pad_token_id: Optional[int] = 0,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model = model

        ############
        ## Declare necessary components

        self.beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            do_early_stopping=False,
            device=device,
        )

        self.logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(min_length, eos_token_id=eos_token_id),
                NoRepeatNGramLogitsProcessor(no_repeat_ngram_size),
            ]
        )

        self.stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=None,
        )

        # instantiate logits processors
        self.logits_warper = self._get_logits_warper(
            top_k=50, top_p=1, temperature=temperature, num_beams=num_beams
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs}

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        encoder_outputs: Any = None,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search with multinomial sampling.

        Parameters:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
        """

        # init values
        output_scores = False
        output_attentions = False

        output_hidden_states = False
        return_dict_in_generate = False
        is_encoder_decoder = False

        if isinstance(model_kwargs, dict):
            model_kwargs["encoder_outputs"] = encoder_outputs
        else:
            model_kwargs = {"encoder_outputs": encoder_outputs}

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        batch_size = len(self.beam_scorer._beam_hyps)
        num_beams = self.beam_scorer.num_beams

        if input_ids is None:
            # init `input_ids` with bos_token_id
            input_ids = torch.full(
                (batch_size * num_beams, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=encoder_outputs.device,
            )

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=encoder_outputs.device
        )
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < self.max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.model(**model_inputs)
            next_token_logits = outputs[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `F.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=self.max_length
            )

            next_token_scores = F.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = self.logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )
            next_token_scores = self.logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = self.beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
            # )

            if self.beam_scorer.is_done:
                break

            if self.stopping_criteria(input_ids, scores):
                break

        sequence_outputs = self.beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


class GeneratorOwn:
    """Implementation of Beam search for inferring process"""

    def __init__(
        self,
        beam_size,
        max_len,
        model,
        init_tok,
        stop_tok,
        no_repeat_ngram_size,
    ):

        self.max_len = max_len
        self.model = model
        self.max_len = max_len
        self.init_tok = init_tok
        self.stop_tok = stop_tok
        self.ngram_nonrepeat = no_repeat_ngram_size
        self.beam_size = beam_size

    def search(self, **kwargs):
        queue = [(0, [self.init_tok]) for _ in range(self.beam_size)]

        for length in range(self.max_len):
            for beamth in range(min(self.beam_size, len(queue))):
                # print(f"Beamth: {beamth}")

                #################################
                # Pop from queue and put into model
                #################################
                accum_prob, beam = queue.pop(0)

                # Pass beam through model
                # Only do this if last token of beam is not self.stop_tok
                if beam[-1] == self.stop_tok:
                    continue
                output = self.model(beam, **kwargs)[-1, :]
                # [d_vocab]

                #################################
                # Apply constraints: no word occurs twice within n-gram,
                # no n-gram occurs twice
                #################################
                disable_words = set()

                # Within n_gram_nonpreeat words, no 2 words are the same
                for n in range(1, min([self.ngram_nonrepeat - 1, len(beam)])):
                    disable_words.add(beam[-n])

                # Form n-1 gram from n - 1 previous words
                if self.ngram_nonrepeat < len(beam):
                    sub_gram = beam[-self.ngram_nonrepeat + 1 :]
                else:
                    sub_gram = None

                # Find all next words of sub_gram in beam
                if sub_gram:
                    list_next = self.find_next(sub_gram, beam)

                    disable_words = disable_words | list_next

                # Disable all words in disable list
                for word in disable_words:
                    output[word] = 0

                #################################
                # Calculate log_softmax and topk
                #################################
                distribution = torch.log_softmax(output, dim=0)
                topk_dist, topk_tok = torch.topk(distribution, self.beam_size, 0)
                # topk_dist, topk_tok: [beam_size]

                # for each dis and token in top-k, create new beam
                for dist_, tok_ in zip(topk_dist, topk_tok):
                    accum_dist_ = accum_prob + dist_.item()
                    beam_ = beam + [tok_.item()]
                    queue.append((accum_dist_, beam_))

            queue = sorted(queue, key=lambda x: x[0], reverse=True)[: self.beam_size]

        return queue[0][1]

    def find_next(self, sub_list: list, main_list: list) -> list:
        """Find all occurences of sub_list in main_list and return the number next to the sub_list in main_list.

        Args:
            sub_list (list): list to check
            main_list (list): list to be checked

        Returns:
            list: list of all next numbers of sub_list in main_list
        """
        sub_ = " ".join(map(str, sub_list))
        main_ = " ".join(map(str, main_list))

        n_num_sub = sub_.count(" ")

        list_next = []
        for m in re.finditer(sub_, main_):
            idx = m.start()

            n_nums_main = main_[:idx].count(" ")

            next_num = n_num_sub + n_nums_main + 1
            if next_num < len(main_list):
                list_next.append(main_list[next_num])

        return set(list_next)
