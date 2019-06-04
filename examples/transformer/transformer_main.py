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
import pickle
import random
import os
import importlib

import torch
from torchtext import data
import texar as tx
from texar.models import Transformer
from texar.utils.utils import adjust_learning_rate

from utils import data_utils, utils
from utils.preprocess import eos_token_id
from bleu_tool import bleu_wrapper

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
    print('logging file is saved in: %s', logging_file)

    model = Transformer(config_model, config_data)
    if torch.cuda.is_available():
        model = model.cuda()

    best_results = {'score': 0, 'epoch': -1}
    opt = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.997), eps=1e-9
    )

    def _eval_epoch(epoch, mode):
        if mode == 'eval':
            eval_data = dev_data
        elif mode == 'test':
            eval_data = test_data
        else:
            raise ValueError('`mode` should be either "eval" or "test".')

        references, hypotheses = [], []
        bsize = config_data.test_batch_size
        for i in range(0, len(eval_data), bsize):
            sources, targets = zip(*eval_data[i:i + bsize])
            x_block = data_utils.source_pad_concat_convert(sources)
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
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)

        if mode == 'eval':
            # Writes results to files to evaluate BLEU
            # For 'eval' mode, the BLEU is based on token ids (rather than
            # text tokens) and serves only as a surrogate metric to monitor
            # the training process
            fname = os.path.join(args.model_dir, 'tmp.eval')
            hypotheses = tx.utils.str_join(hypotheses)
            references = tx.utils.str_join(references)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hypotheses, references, fname, mode='s')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info('epoch: %d, eval_bleu %.4f', epoch, eval_bleu)
            print('epoch: %d, eval_bleu %.4f' % (epoch, eval_bleu))

            if eval_bleu > best_results['score']:
                logger.info('epoch: %d, best bleu: %.4f', epoch, eval_bleu)
                best_results['score'] = eval_bleu
                best_results['epoch'] = epoch
                model_path = os.path.join(args.model_dir, args.model_fn)
                logger.info('saving model to %s', model_path)
                print('saving model to %s' % model_path)

                states = {
                    'model': model.state_dict(),
                    'optimizer': model.state_dict()
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
            logger.info('Test output writtn to file: %s', hyp_fn)
            print('Test output writtn to file: %s' % hyp_fn)

    def _train_epoch(epoch: int):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            batch_size_fn=utils.batch_size_fn,
            random_shuffler=data.iterator.RandomShuffler())

        for _, train_batch in enumerate(train_iter):
            opt.zero_grad()
            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)
            loss = model(
                encoder_input=in_arrays[0],
                is_train_mode=True,
                decoder_input=in_arrays[1],
                labels=in_arrays[2],
            )
            loss.backward()
            lr = utils.get_lr(model.step_iteration, config_model.lr_config)
            adjust_learning_rate(opt, lr)
            opt.step()
            model.step_iteration += 1
            step = model.step_iteration
            if step % config_data.display_steps == 0:
                logger.info('step: %d, loss: %.4f', step, loss)
                print('lr: {} step: {}, loss: {}'.format(lr, step, loss))
            if step and step % config_data.eval_steps == 0:
                _eval_epoch(epoch, mode='eval')

    if args.run_mode == 'train_and_evaluate':
        logger.info('Begin running with train_and_evaluate mode')
        model_path = os.path.join(args.model_dir, args.model_fn)
        if os.path.exists(model_path):
            logger.info('Restore latest checkpoint in %s' % model_path)
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['optimizer'])
            _eval_epoch(0, mode='test')

        for epoch in range(config_data.max_train_epoch):
            _train_epoch(epoch)

    elif args.run_mode == 'test':
        logger.info('Begin running with test mode')
        model_path = os.path.join(args.model_dir, args.model_fn)
        logger.info('Restore latest checkpoint in %s' % model_path)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        _eval_epoch(0, mode='test')

    else:
        raise ValueError('Unknown mode: {}'.format(args.run_mode))


if __name__ == '__main__':
    main()
