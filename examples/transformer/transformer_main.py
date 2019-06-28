# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer model.
"""
from typing import Optional

import argparse
import functools
import importlib
import os
import random
import torch
from torch import nn
from torchtext import data
from tqdm import tqdm

import texar as tx
from texar.data import Vocab
from texar.module_base import ModuleBase
from texar.modules import TransformerDecoder
from texar.modules import WordEmbedder
from texar.modules import SinusoidsPositionEmbedder
from texar.modules import TransformerEncoder
from texar.losses import sequence_softmax_cross_entropy

from bleu_tool import bleu_wrapper
from utils import data_utils, utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config_model", type=str, default="config_model", help="The model config."
)
parser.add_argument(
    "--config_data",
    type=str,
    default="config_iwslt15",
    help="The dataset config.",
)
parser.add_argument(
    "--run_mode",
    type=str,
    default="train_and_evaluate",
    help="Either train_and_evaluate or evaluate or test.",
)
parser.add_argument(
    "--model_dir",
    type=str,
    default="./outputs/",
    help="Path to save the trained model and logs.",
)
parser.add_argument(
    "--model_fn",
    type=str,
    default="best-model.ckpt",
    help="Model filename to save the trained weights",
)

args = parser.parse_args()

config_model = importlib.import_module(args.config_model)
config_data = importlib.import_module(args.config_data)

utils.set_random_seed(config_model.random_seed)


class Transformer(ModuleBase):
    r"""A standalone sequence-to-sequence Transformer model.
    TODO: Add detailed docstrings.
    """

    def __init__(self, model_config, data_config, vocab: Vocab):
        ModuleBase.__init__(self)

        self.config_model = model_config
        self.config_data = data_config
        self.vocab = vocab
        self.vocab_size = vocab.size

        self.word_embedder = WordEmbedder(
            vocab_size=self.vocab_size, hparams=config_model.emb
        )
        self.pos_embedder = SinusoidsPositionEmbedder(
            position_size=config_data.max_decoding_length,
            hparams=config_model.position_embedder_hparams,
        )

        self.encoder = TransformerEncoder(hparams=config_model.encoder)
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            output_layer=self.word_embedder.embedding,
            hparams=config_model.decoder,
        )

        self.smoothed_loss_func = LabelSmoothingLoss(
            label_confidence=config_model.loss_label_confidence,
            tgt_vocab_size=self.vocab_size,
            ignore_index=0,
        )

    def forward(  # type: ignore
        self,
        encoder_input: torch.Tensor,
        is_train_mode: Optional[bool],
        decoder_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        beam_width: Optional[int] = None,
    ):
        r"""TODO: Add detailed docstrings.

        Args:
            encoder_input:
            is_train_mode:
            decoder_input:
            labels:
            beam_width:

        Returns:

        """

        batch_size = encoder_input.size()[0]
        # (text sequence length excluding padding)
        encoder_input_length = (encoder_input != 0).int().sum(dim=1)

        if is_train_mode:
            self.train()

        else:
            self.eval()

        # Source word embedding
        src_word_embeds = self.word_embedder(encoder_input)
        src_word_embeds = src_word_embeds * config_model.hidden_dim ** 0.5

        # Position embedding (shared b/w source and target)
        src_seq_len = torch.full(
            (batch_size,), encoder_input.size()[1], dtype=torch.int32,
            device=encoder_input.device
        )

        src_pos_embeds = self.pos_embedder(sequence_length=src_seq_len)
        src_input_embedding = src_word_embeds + src_pos_embeds

        encoder_output = self.encoder(
            inputs=src_input_embedding, sequence_length=encoder_input_length
        )

        if is_train_mode:
            assert decoder_input is not None
            assert labels is not None

            tgt_word_embeds = self.word_embedder(decoder_input)
            tgt_word_embeds = (
                tgt_word_embeds * config_model.hidden_dim ** 0.5
            )
            tgt_seq_len = torch.full(
                (batch_size,), decoder_input.size()[1], dtype=torch.int32
            )
            tgt_seq_len = tgt_seq_len.to(device=decoder_input.device)

            tgt_pos_embeds = self.pos_embedder(sequence_length=tgt_seq_len)

            tgt_input_embedding = tgt_word_embeds + tgt_pos_embeds

            # For training
            outputs = self.decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                inputs=tgt_input_embedding,
                decoding_strategy="train_greedy",
            )
            labels = labels.to(device=outputs.logits.device)
            label_lengths = (labels != 0).long().sum(dim=1)
            label_lengths = label_lengths.to(device=outputs.logits.device)
            is_target = (labels != 0).float()
            mle_loss = self.smoothed_loss_func(
                outputs.logits, labels, label_lengths
            )
            mle_loss = (mle_loss * is_target).sum() / is_target.sum()
            return mle_loss
        else:
            start_tokens = encoder_input.new_full(
                (batch_size,), self.vocab.bos_token_id, dtype=torch.long
            )

            def _embedding_fn(x, y):
                word_embed = self.word_embedder(x)
                scale = config_model.hidden_dim ** 0.5
                pos_embed = self.pos_embedder(y)
                return word_embed * scale + pos_embed

            predictions = self.decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                beam_width=beam_width,
                length_penalty=config_model.length_penalty,
                start_tokens=start_tokens,
                end_token=self.vocab.eos_token_id,
                embedding=_embedding_fn,
                max_decoding_length=config_data.max_decoding_length,
                decoding_strategy="infer_greedy",
            )
            # Uses the best sample by beam search
            return predictions


class LabelSmoothingLoss(nn.Module):
    r"""With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.

    Args:
        label_confidence: the confidence weight on the ground truth label.
        tgt_vocab_size: the size of the final classification.
        ignore_index: The index in the vocabulary to ignore weight.
    """

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size
        super().__init__()

        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(  # type: ignore
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        label_lengths: torch.LongTensor,
    ) -> torch.Tensor:
        r"""

        Args:
            output (FloatTensor): batch_size x seq_length * n_classes
            target (LongTensor): batch_size * seq_length, specify the label
                target
            label_lengths(torch.LongTensor): specify the length of the labels
        """
        ori_shapes = (output.size(), target.size())
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        output = output.view(ori_shapes[0])
        model_prob = model_prob.view(ori_shapes[0])

        return sequence_softmax_cross_entropy(
            labels=model_prob,
            logits=output,
            sequence_length=label_lengths,
            average_across_batch=False,
            sum_over_timesteps=False,
        )


def main():
    """Entry point.
    """
    # Load data
    train_data, dev_data, test_data = data_utils.load_data_numpy(
        config_data.input_dir, config_data.filename_prefix
    )

    vocab = Vocab(config_data.vocab_file)

    beam_width = config_model.beam_width

    # Create logging
    tx.utils.maybe_create_dir(args.model_dir)
    logging_file = os.path.join(args.model_dir, "logging.txt")
    logger = utils.get_logger(logging_file)
    print(f"logging file is saved in: {logging_file}")

    model = Transformer(config_model, config_data, vocab)
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.cuda.current_device()
    else:
        device = None

    best_results = {"score": 0, "epoch": -1}
    lr_config = config_model.lr_config
    if lr_config["learning_rate_schedule"] == "static":
        init_lr = lr_config["static_lr"]
        scheduler_lambda = lambda x: 1.0
    else:
        init_lr = lr_config["lr_constant"]
        scheduler_lambda = functools.partial(
            utils.get_lr_multiplier, warmup_steps=lr_config["warmup_steps"]
        )
    optim = torch.optim.Adam(
        model.parameters(), lr=init_lr, betas=(0.9, 0.997), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)

    def _eval_epoch(epoch, mode):

        if mode == "eval":
            eval_data = dev_data
        elif mode == "test":
            eval_data = test_data
        else:
            raise ValueError('`mode` should be either "eval" or "test".')
        model.eval()
        references, hypotheses = [], []
        bsize = config_data.test_batch_size
        for i in tqdm(range(0, len(eval_data), bsize)):
            sources, targets = zip(*eval_data[i: i + bsize])
            with torch.no_grad():
                x_block = data_utils.source_pad_concat_convert(
                    sources, device=device
                )
                predictions = model(
                    encoder_input=x_block,
                    is_train_mode=False,
                    beam_width=beam_width,
                )
                if beam_width == 1:
                    decoded_ids = predictions[0].sample_id
                else:
                    decoded_ids = predictions["sample_id"][:, :, 0]

                hypotheses.extend(h.tolist() for h in decoded_ids)
                references.extend(r.tolist() for r in targets)
                hypotheses = utils.list_strip_eos(
                    hypotheses, vocab.eos_token_id
                )
                references = utils.list_strip_eos(
                    references, vocab.eos_token_id
                )

        if mode == "eval":
            # Writes results to files to evaluate BLEU
            # For 'eval' mode, the BLEU is based on token ids (rather than
            # text tokens) and serves only as a surrogate metric to monitor
            # the training process
            # TODO: Use texar.evals.bleu
            fname = os.path.join(args.model_dir, "tmp.eval")
            hwords, rwords = [], []
            for hyp, ref in zip(hypotheses, references):
                hwords.append([str(y) for y in hyp])
                rwords.append([str(y) for y in ref])
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hwords,
                rwords,
                fname,
                mode="s",
                src_fname_suffix="hyp",
                tgt_fname_suffix="ref",
            )
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100.0 * eval_bleu
            logger.info("epoch: %d, eval_bleu %.4f", epoch, eval_bleu)
            print(f"epoch: {epoch:d}, eval_bleu {eval_bleu:.4f}")

            if eval_bleu > best_results["score"]:
                logger.info("epoch: %d, best bleu: %.4f", epoch, eval_bleu)
                best_results["score"] = eval_bleu
                best_results["epoch"] = epoch
                model_path = os.path.join(args.model_dir, args.model_fn)
                logger.info("Saving model to %s", model_path)
                print(f"Saving model to {model_path}")

                states = {
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(states, model_path)

        elif mode == "test":
            # For 'test' mode, together with the cmds in README.md, BLEU
            # is evaluated based on text tokens, which is the standard metric.
            fname = os.path.join(args.model_dir, "test.output")
            hwords, rwords = [], []
            for hyp, ref in zip(hypotheses, references):
                hwords.append([vocab.id_to_token_map_py[y] for y in hyp])
                rwords.append([vocab.id_to_token_map_py[y] for y in ref])
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hwords,
                rwords,
                fname,
                mode="s",
                src_fname_suffix="hyp",
                tgt_fname_suffix="ref",
            )
            logger.info("Test output written to file: %s", hyp_fn)
            print(f"Test output written to file: {hyp_fn}")

    def _train_epoch(epoch: int):
        random.shuffle(train_data)
        model.train()
        train_iter = data.iterator.pool(
            train_data,
            config_data.batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            # key is not used if sort_within_batch is False by default
            batch_size_fn=utils.batch_size_fn,
            random_shuffler=data.iterator.RandomShuffler(),
        )

        for _, train_batch in tqdm(enumerate(train_iter)):
            optim.zero_grad()
            in_arrays = data_utils.seq2seq_pad_concat_convert(
                train_batch, device=device
            )
            loss = model(
                encoder_input=in_arrays[0],
                is_train_mode=True,
                decoder_input=in_arrays[1],
                labels=in_arrays[2],
            )
            loss.backward()

            optim.step()
            scheduler.step()

            step = scheduler.last_epoch
            if step % config_data.display_steps == 0:
                logger.info("step: %d, loss: %.4f", step, loss)
                lr = optim.param_groups[0]["lr"]
                print(f"lr: {lr} step: {step}, loss: {loss:.4}")
            if step and step % config_data.eval_steps == 0:
                _eval_epoch(epoch, mode="eval")

    if args.run_mode == "train_and_evaluate":
        logger.info("Begin running with train_and_evaluate mode")
        model_path = os.path.join(args.model_dir, args.model_fn)
        if os.path.exists(model_path):
            logger.info("Restore latest checkpoint in", model_path)
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            _eval_epoch(0, mode="test")

        for epoch in range(config_data.max_train_epoch):
            _train_epoch(epoch)
            _eval_epoch(epoch, mode="eval")

    elif args.run_mode == "evaluate":
        logger.info("Begin running with evaluate mode")
        model_path = os.path.join(args.model_dir, args.model_fn)
        logger.info("Restore latest checkpoint in %s", model_path)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt["model"])
        _eval_epoch(0, mode="eval")

    elif args.run_mode == "test":
        logger.info("Begin running with test mode")
        model_path = os.path.join(args.model_dir, args.model_fn)
        logger.info("Restore latest checkpoint in", model_path)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt["model"])
        _eval_epoch(0, mode="test")

    else:
        raise ValueError(f"Unknown mode: {args.run_mode}")


if __name__ == "__main__":
    main()
