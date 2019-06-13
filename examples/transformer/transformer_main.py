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
import pickle
import random
from tqdm import tqdm

import torch
from torchtext import data

import texar as tx
from bleu_tool import bleu_wrapper
from texar.modules import Transformer
from utils import data_utils, utils
from utils.preprocess import eos_token_id

# pylint: disable=invalid-name, too-many-locals

parser = argparse.ArgumentParser()

parser.add_argument("--config_model",
                    type=str,
                    default="config_model",
                    help="The model config.")
parser.add_argument("--config_data",
                    type=str,
                    default="config_iwslt15",
                    help="The dataset config.")
parser.add_argument("--run_mode",
                    type=str,
                    default="train_and_evaluate",
                    help="Either train_and_evaluate or test.")
parser.add_argument("--model_dir",
                    type=str,
                    default="./outputs/",
                    help="Path to save the trained model and logs.")
parser.add_argument("--model_fn",
                    type=str,
                    default="best-model.ckpt",
                    help="Model filename to save the trained weights")

args = parser.parse_args()

config_model = importlib.import_module(args.config_model)
config_data = importlib.import_module(args.config_data)

utils.set_random_seed(config_model.random_seed)


def main():
    """Entry point.
    """
    # Load data
    train_data, dev_data, test_data = data_utils.load_data_numpy(
        config_data.input_dir, config_data.filename_prefix)
    with open(config_data.vocab_file, 'rb') as f:
        id2w = pickle.load(f)
    vocab_size = len(id2w)

    beam_width = getattr(config_model, "beam_width", 1)

    # Create logging
    tx.utils.maybe_create_dir(args.model_dir)
    logging_file = os.path.join(args.model_dir, 'logging.txt')
    logger = utils.get_logger(logging_file)
    print(f"logging file is saved in: {logging_file}")

    model = Transformer(config_model, config_data)
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.cuda.current_device()
    else:
        device = None

    best_results = {'score': 0, 'epoch': -1}
    lr_config = config_model.lr_config
    if lr_config["learning_rate_schedule"] == "static":
        init_lr = lr_config["static_lr"]
        scheduler_lambda = lambda x: 1.0
    else:
        init_lr = lr_config["lr_constant"]
        scheduler_lambda = functools.partial(
            utils.get_lr_multiplier, warmup_steps=lr_config["warmup_steps"])
    optim = torch.optim.Adam(
        model.parameters(), lr=init_lr, betas=(0.9, 0.997), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)

    def _eval_epoch(epoch, mode):
        torch.cuda.empty_cache()
        if mode == 'eval':
            eval_data = dev_data
        elif mode == 'test':
            eval_data = test_data
        else:
            raise ValueError("`mode` should be either \"eval\" or \"test\".")

        references, hypotheses = [], []
        bsize = config_data.test_batch_size
        for i in tqdm(range(0, len(eval_data), bsize)):
            sources, targets = zip(*eval_data[i:i + bsize])
            with torch.no_grad():
                x_block = data_utils.source_pad_concat_convert(
                    sources, device=device)
                predictions = model(
                    encoder_input=x_block,
                    is_train_mode=False,
                    beam_width=beam_width)
                if beam_width == 1:
                    decoded_ids = predictions[0].sample_id
                else:
                    decoded_ids = predictions["sample_id"][:, :, 0]

                hypotheses.extend(h.tolist() for h in decoded_ids)
                references.extend(r.tolist() for r in targets)
                hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
                references = utils.list_strip_eos(references, eos_token_id)

        if mode == 'eval':
            # Writes results to files to evaluate BLEU
            # For 'eval' mode, the BLEU is based on token ids (rather than
            # text tokens) and serves only as a surrogate metric to monitor
            # the training process
            fname = os.path.join(args.model_dir, 'tmp.eval')
            hwords, rwords = [], []
            for hyp, ref in zip(hypotheses, references):
                hwords.append([str(y) for y in hyp])
                rwords.append([str(y) for y in ref])
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hwords, rwords, fname, mode='s',
                src_fname_suffix='hyp', tgt_fname_suffix='ref')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info("epoch: %d, eval_bleu %.4f", epoch, eval_bleu)
            print(f"epoch: {epoch:d}, eval_bleu {eval_bleu:.4f}")

            if eval_bleu > best_results['score']:
                logger.info("epoch: %d, best bleu: %.4f", epoch, eval_bleu)
                best_results['score'] = eval_bleu
                best_results['epoch'] = epoch
                model_path = os.path.join(args.model_dir, args.model_fn)
                logger.info("Saving model to %s", model_path)
                print(f"Saving model to {model_path}")

                states = {
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(states, model_path)

        elif mode == 'test':
            # For 'test' mode, together with the cmds in README.md, BLEU
            # is evaluated based on text tokens, which is the standard metric.
            fname = os.path.join(args.model_dir, 'test.output')
            hwords, rwords = [], []
            for hyp, ref in zip(hypotheses, references):
                hwords.append([id2w[y] for y in hyp])
                rwords.append([id2w[y] for y in ref])
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hwords, rwords, fname, mode='s',
                src_fname_suffix='hyp', tgt_fname_suffix='ref')
            logger.info("Test output written to file: %s", hyp_fn)
            print(f"Test output written to file: {hyp_fn}")

    def _train_epoch(epoch: int):
        torch.cuda.empty_cache()
        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            # key is not used if sort_within_batch is False by default
            batch_size_fn=utils.batch_size_fn,
            random_shuffler=data.iterator.RandomShuffler())

        for _, train_batch in tqdm(enumerate(train_iter)):
            optim.zero_grad()
            in_arrays = data_utils.seq2seq_pad_concat_convert(
                train_batch, device=device)
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
                logger.info('step: %d, loss: %.4f', step, loss)
                lr = optim.param_groups[0]['lr']
                print(f"lr: {lr} step: {step}, loss: {loss:.4}")
            if step and step % config_data.eval_steps == 0:
                _eval_epoch(epoch, mode='eval')

    if args.run_mode == 'train_and_evaluate':
        logger.info("Begin running with train_and_evaluate mode")
        model_path = os.path.join(args.model_dir, args.model_fn)
        if os.path.exists(model_path):
            logger.info("Restore latest checkpoint in %s", model_path)
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            _eval_epoch(0, mode='test')

        for epoch in range(config_data.max_train_epoch):
            _train_epoch(epoch)
            _eval_epoch(epoch, mode='eval')

    elif args.run_mode == 'eval':
        logger.info("Begin running with evaluate mode")
        model_path = os.path.join(args.model_dir, args.model_fn)
        logger.info("Restore latest checkpoint in %s", model_path)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model'])
        _eval_epoch(0, mode='eval')

    elif args.run_mode == 'test':
        logger.info("Begin running with test mode")
        model_path = os.path.join(args.model_dir, args.model_fn)
        logger.info("Restore latest checkpoint in %s", model_path)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model'])
        _eval_epoch(0, mode='test')

    else:
        raise ValueError(f"Unknown mode: {args.run_mode}")


if __name__ == '__main__':
    main()
