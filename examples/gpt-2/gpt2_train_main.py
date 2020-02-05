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
"""Example of fine-tuning OpenAI GPT-2 language model.
"""

import argparse
import importlib
import os
from typing import Any

import torch
import texar.torch as tx

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint', type=str, default=None,
    help="Model checkpoint to load model weights from.")
parser.add_argument(
    "--pretrained-model-name", type=str, default="gpt2-small",
    choices=tx.modules.GPT2Decoder.available_checkpoints(),
    help="Name of the pre-trained checkpoint to load.")
parser.add_argument(
    '--config-train', type=str, default="config_train",
    help="Configurations of GPT-2 training, including data and "
         "optimization hyperparameters.")
parser.add_argument(
    "--output-dir", default="output/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    '--temperature', type=float, default=0.7,
    help="Softmax temperature for top-k sample decoding. Must be strictly "
         "greater than 0. Defaults to 0.7.")
parser.add_argument(
    '--top-k', type=int, default=40,
    help="The number of top most likely candidates from a vocab distribution.")
parser.add_argument(
    '--top-p', type=float, default=None,
    help="Select tokens with cumulative probability of at most 'p' when "
         "arranged in decreasing order. This will use "
         "TopPSampleEmbeddingHelper for decoding.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do-test", action="store_true",
    help="Whether to run test on the test set.")

args = parser.parse_args()

config_train: Any = importlib.import_module(args.config_train)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(args.output_dir)

    max_decoding_length = config_train.max_decoding_length

    # Build the GPT-2 model
    model = tx.modules.GPT2Decoder(args.pretrained_model_name)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
    model.to(device)

    if max_decoding_length > model.hparams.position_size:
        raise ValueError(
            "max_decoding_length should not be greater than position size")

    # Create a GPT-2 tokenizer (BPE encoding)
    tokenizer = tx.data.GPT2Tokenizer(
        pretrained_model_name=args.pretrained_model_name)

    # Loads data
    datasets = {}
    if args.do_train:
        train_dataset = tx.data.RecordData(
            hparams=config_train.train_hparam, device=device)
        datasets['train'] = train_dataset
    if args.do_eval:
        eval_dataset = tx.data.RecordData(
            hparams=config_train.eval_hparam, device=device)
        datasets['eval'] = eval_dataset
    if args.do_test:
        test_dataset = tx.data.RecordData(
            hparams=config_train.test_hparam, device=device)
        datasets['test'] = test_dataset
    iterator = tx.data.DataIterator(datasets)

    # For training
    train_op = tx.core.get_train_op(
        params=model.parameters(), hparams=config_train.opt)

    end_token = tokenizer.map_token_to_id('<|endoftext|>')

    def _get_helper(start_tokens):
        if args.top_p:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=end_token,
                p=args.top_p,
                softmax_temperature=args.temperature)
        else:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=end_token,
                top_k=args.top_k,
                softmax_temperature=args.temperature)
        return helper

    dis_steps = config_train.display_steps
    eval_steps = config_train.eval_steps

    eval_best = {"loss": 1e8, "ppl": 1e8}

    def _train_epoch():
        r"""Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.switch_to_dataset("train")
        model.train()

        step = 0
        for batch in iterator:
            input_ids = batch["text_ids"]

            outputs = model(inputs=input_ids, decoding_strategy='train_greedy')

            loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['text_ids'][:, 1:],
                logits=outputs.logits[:, :-1, :],
                sequence_length=batch['length'] - 1,
                average_across_timesteps=True,
                sum_over_timesteps=False)
            loss.backward()
            train_op()

            if dis_steps > 0 and step % dis_steps == 0:
                print("step={}, loss={:.4f}".format(step, loss))

            if eval_steps > 0 and step % eval_steps == 0:
                _eval_epoch()

            step += 1

    @torch.no_grad()
    def _eval_epoch():
        r"""Evaluates on the dev set.
        """
        iterator.switch_to_dataset("eval")
        model.eval()

        nsamples = 0
        avg_rec = tx.utils.AverageRecorder()
        for batch in iterator:
            input_ids = batch["text_ids"]

            outputs = model(inputs=input_ids)

            loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['text_ids'][:, 1:],
                logits=outputs.logits[:, :-1, :],
                sequence_length=batch['length'] - 1,
                average_across_timesteps=True,
                sum_over_timesteps=False)
            ppl = torch.exp(loss)
            batch_size = input_ids.size()[0]
            avg_rec.add([loss, ppl], batch_size)
            nsamples += batch_size

        print("eval loss: {:.4f}; ppl: {:.4f}; "
              "nsamples: {:d}".format(avg_rec.avg(0), avg_rec.avg(1), nsamples))

        if args.do_train and avg_rec.avg(0) < eval_best["loss"]:
            eval_best["loss"] = avg_rec.avg(0)
            eval_best["ppl"] = avg_rec.avg(1)
            ckpt_fn = os.path.join(args.output_dir, 'model_best.ckpt')
            torch.save(model.state_dict(), ckpt_fn)
            print("Checkpoint best to {}".format(ckpt_fn))

    @torch.no_grad()
    def _test_epoch():
        r"""Generates samples on the test set.
        """
        iterator.switch_to_dataset("test")
        model.eval()

        _all_inputs = []
        _all_samples = []

        for batch in iterator:
            input_ids = batch["text_ids"]
            length = batch["length"]
            start_tokens = input_ids[:, 0]
            helper = _get_helper(start_tokens)

            output, _ = model(
                context=input_ids,
                context_sequence_length=length,
                max_decoding_length=max_decoding_length,
                helper=helper)
            sample_id = output.sample_id

            _inputs = []
            for i, l in zip(input_ids, length):
                # Delete padding
                _inputs.append(i[:l].tolist())
            _all_inputs.extend(_inputs)

            _samples = []
            for s, l in zip(sample_id, length):
                # Delte inputs from samples
                _samples.append(s[l:].tolist())
            _all_samples.extend(_samples)

        # Parse samples and write to file

        eos_token_id = tokenizer.map_token_to_id('<|endoftext|>')

        _all_input_text = []
        for i in _all_inputs:
            if i[0] == eos_token_id:
                # '<|endoftext|>' is used as the BOS token. Delete it here
                i = i[1:]
            i_text = tokenizer.map_id_to_text(i)
            _all_input_text.append(i_text)
        # '<|endoftext|>' is used as the PAD token. Delete them here
        _all_input_text = tx.utils.strip_eos(_all_input_text,
                                             eos_token='<|endoftext|>')

        _all_samples_text = []
        for i, s in zip(_all_inputs, _all_samples):
            s_text = tokenizer.map_id_to_text(s)
            s_text = s_text.replace('\n', ' ')
            _all_samples_text.append(s_text)
        _all_samples_text = tx.utils.strip_eos(_all_samples_text,
                                               eos_token='<|endoftext|>')

        output_file = os.path.join(args.output_dir, "test_samples.tsv")
        print('Write samples to {}'.format(output_file))
        tx.utils.write_paired_text(
            _all_input_text, _all_samples_text, output_file)

    if args.do_train:
        for _ in range(config_train.max_train_epoch):
            _train_epoch()
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, 'model.ckpt'))

    if args.do_eval:
        _eval_epoch()

    if args.do_test:
        _test_epoch()


if __name__ == "__main__":
    main()
