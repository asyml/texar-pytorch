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

import argparse
import functools
import importlib
import os
from typing import Any

import torch
import tqdm
import texar.torch as tx

from model import Transformer
import utils.data_utils as data_utils
import utils.utils as utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-model", type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    "--config-data", type=str, default="config_iwslt15",
    help="The dataset config.")
parser.add_argument(
    "--run-mode", type=str, default="train_and_evaluate",
    help="Either train_and_evaluate or evaluate or test.")
parser.add_argument(
    "--output-dir", type=str, default="./outputs/",
    help="Path to save the trained model and logs.")
parser.add_argument(
    "--output-filename", type=str, default="best-model.ckpt",
    help="Model filename to save the trained weights")

args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)

utils.set_random_seed(config_model.random_seed)


def main():
    """Entry point.
    """
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
        print(f"Using CUDA device {device}")
    else:
        device = None

    # Load data
    vocab = tx.data.Vocab(config_data.vocab_file)
    data_hparams = {
        # "batch_size" is ignored for train since we use dynamic batching.
        "batch_size": config_data.test_batch_size,
        "pad_id": vocab.pad_token_id,
        "bos_id": vocab.bos_token_id,
        "eos_id": vocab.eos_token_id,
    }
    datasets = {
        split: data_utils.Seq2SeqData(
            os.path.join(
                config_data.input_dir,
                f"{config_data.filename_prefix}{split}.npy"
            ),
            # Only shuffle during training.
            hparams={**data_hparams, "shuffle": split == "train"},
            device=device,
        ) for split in ["train", "valid", "test"]
    }
    print(f"Training data size: {len(datasets['train'])}")
    beam_width = config_model.beam_width

    # Create logging
    tx.utils.maybe_create_dir(args.output_dir)
    logging_file = os.path.join(args.output_dir, "logging.txt")
    logger = utils.get_logger(logging_file)
    print(f"logging file is saved in: {logging_file}")

    # Create model and optimizer
    model = Transformer(config_model, config_data, vocab).to(device)

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

    @torch.no_grad()
    def _eval_epoch(epoch, mode, print_fn=None):
        if print_fn is None:
            print_fn = print
            tqdm_leave = True
        else:
            tqdm_leave = False
        model.eval()
        eval_data = datasets[mode]
        eval_iter = tx.data.DataIterator(eval_data)
        references, hypotheses = [], []
        for batch in tqdm.tqdm(eval_iter, ncols=80, leave=tqdm_leave,
                               desc=f"Eval on {mode} set"):
            predictions = model(
                encoder_input=batch.source,
                beam_width=beam_width,
            )
            if beam_width == 1:
                decoded_ids = predictions[0].sample_id
            else:
                decoded_ids = predictions["sample_id"][:, :, 0]

            hypotheses.extend(h.tolist() for h in decoded_ids)
            references.extend(r.tolist() for r in batch.target_output)
        hypotheses = utils.list_strip_eos(hypotheses, vocab.eos_token_id)
        references = utils.list_strip_eos(references, vocab.eos_token_id)

        if mode == "valid":
            # Writes results to files to evaluate BLEU
            # For 'eval' mode, the BLEU is based on token ids (rather than
            # text tokens) and serves only as a surrogate metric to monitor
            # the training process
            fname = os.path.join(args.output_dir, "tmp.eval")
            hwords, rwords = [], []
            for hyp, ref in zip(hypotheses, references):
                hwords.append([str(y) for y in hyp])
                rwords.append([str(y) for y in ref])
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_file, ref_file = tx.utils.write_paired_text(
                hwords, rwords, fname, mode="s",
                src_fname_suffix="hyp", tgt_fname_suffix="ref",
            )
            eval_bleu = tx.evals.file_bleu(ref_file, hyp_file,
                                           case_sensitive=True)
            logger.info("epoch: %d, eval_bleu %.4f", epoch, eval_bleu)
            print_fn(f"epoch: {epoch:d}, eval_bleu {eval_bleu:.4f}")

            if eval_bleu > best_results["score"]:
                logger.info("epoch: %d, best bleu: %.4f", epoch, eval_bleu)
                best_results["score"] = eval_bleu
                best_results["epoch"] = epoch
                model_path = os.path.join(args.output_dir, args.output_filename)
                logger.info("Saving model to %s", model_path)
                print_fn(f"Saving model to {model_path}")

                states = {
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(states, model_path)

        elif mode == "test":
            # For 'test' mode, together with the commands in README.md, BLEU
            # is evaluated based on text tokens, which is the standard metric.
            fname = os.path.join(args.output_dir, "test.output")
            hwords, rwords = [], []
            for hyp, ref in zip(hypotheses, references):
                hwords.append(vocab.map_ids_to_tokens_py(hyp))
                rwords.append(vocab.map_ids_to_tokens_py(ref))
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_file, ref_file = tx.utils.write_paired_text(
                hwords, rwords, fname, mode="s",
                src_fname_suffix="hyp", tgt_fname_suffix="ref",
            )
            logger.info("Test output written to file: %s", hyp_file)
            print_fn(f"Test output written to file: {hyp_file}")

    def _train_epoch(epoch: int):
        model.train()
        train_iter = tx.data.DataIterator(
            datasets["train"],
            data_utils.CustomBatchingStrategy(config_data.max_batch_tokens)
        )

        progress = tqdm.tqdm(
            train_iter, ncols=80,
            desc=f"Training epoch {epoch}",
        )
        for train_batch in progress:
            optim.zero_grad()
            loss = model(
                encoder_input=train_batch.source,
                decoder_input=train_batch.target_input,
                labels=train_batch.target_output,
            )
            loss.backward()

            optim.step()
            scheduler.step()

            step = scheduler.last_epoch
            if step % config_data.display_steps == 0:
                logger.info("step: %d, loss: %.4f", step, loss)
                lr = optim.param_groups[0]["lr"]
                progress.write(f"lr: {lr:.4e} step: {step}, loss: {loss:.4}")
            if step and step % config_data.eval_steps == 0:
                _eval_epoch(epoch, mode="valid", print_fn=progress.write)
        progress.close()

    model_path = os.path.join(args.output_dir, args.output_filename)

    if args.run_mode == "train_and_evaluate":
        logger.info("Begin running with train_and_evaluate mode")
        if os.path.exists(model_path):
            logger.info("Restore latest checkpoint in %s", model_path)
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            _eval_epoch(0, mode="valid")

        for epoch in range(config_data.max_train_epoch):
            _train_epoch(epoch)
            _eval_epoch(epoch, mode="valid")

    elif args.run_mode in ["evaluate", "test"]:
        logger.info("Begin running with %s mode", args.run_mode)
        logger.info("Restore latest checkpoint in %s", model_path)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt["model"])
        _eval_epoch(0, mode=("test" if args.run_mode == "test" else "valid"))

    else:
        raise ValueError(f"Unknown mode: {args.run_mode}")


if __name__ == "__main__":
    main()
