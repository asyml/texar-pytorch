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
"""Example of building OpenAI GPT-2 language model for sample generation.
"""
import os
import argparse
import importlib
import random

import numpy as np

import torch

import sys
sys.path.append("/Users/pengzhi.gao/Desktop/my_git/texar-pytorch")

import texar as tx

from utils import processor


parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint', type=str, default=None,
    help="Model checkpoint to load model weights from.")
parser.add_argument(
    '--config_model', type=str, default="configs.config_model_117M",
    help="The model configuration file to configure the model.")
parser.add_argument(
    '--seed', type=int, default=None, help="Random seed.")
parser.add_argument(
    '--nsamples', type=int, default=1, help="The number of samples per input.")
parser.add_argument(
    '--batch_size', type=int, default=1, help="The batch size of input.")
parser.add_argument(
    '--max_decoding_length', type=int, default=128,
    help="The maximun length of generated text.")
parser.add_argument(
    '--temperature', type=float, default=0.7,
    help="Softmax temperature for top-k sample decoding. Must be strictly "
         "greater than 0. Defaults to 0.7.")
parser.add_argument(
    '--top_k', type=int, default=40,
    help="The number of top most likely candidates from a vocab distribution.")
parser.add_argument(
    '--p', type=int, default=None,
    help="Select tokens with cumulative probability of at most 'p' when "
         "arranged in decreasing order. This will use "
         "TopPSampleEmbeddingHelper for decoding.")
parser.add_argument(
    '--is_interactive', action='store_true', help="Interactive mode or not.")

args = parser.parse_args()

config_model = importlib.import_module(args.config_model)
config_model = {
    k: v for k, v in config_model.__dict__.items()
    if not k.startswith('__')}
config_model.pop("dim")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Builds the model and runs.
    """
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    nsamples = args.nsamples
    batch_size = args.batch_size
    max_decoding_length = args.max_decoding_length

    assert max_decoding_length <= config_model["position_size"], (
        "max_decoding_length should not be greater than position size")
    assert nsamples % batch_size == 0, (
        "nsamples must be dividable by batch_size")

    # Build the GPT-2 model
    model = tx.modules.GPT2Decoder(cache_dir='gpt2_pretrained_models',
                                   hparams=config_model)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
    model.to(device)

    # Create a data pre-processor for, e.g., BPE encoding
    proc = processor.get_encoder(os.path.join(
        'gpt2_pretrained_models', config_model["pretrained_model_name"]))
    end_token = proc.encoder['<|endoftext|>']

    print("\nFinished loading\n")

    _embedding_fn = lambda x, y: (
            model.word_embedder(x) + model.position_embedder(y))

    def _get_helper(start_tokens):
        if args.p:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                embedding=_embedding_fn,
                start_tokens=start_tokens,
                end_token=end_token,
                p=args.p,
                softmax_temperature=args.temperature)
        else:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                embedding=_embedding_fn,
                start_tokens=start_tokens,
                end_token=end_token,
                top_k=args.top_k,
                softmax_temperature=args.temperature)
        return helper

    if args.is_interactive:
        # Generate continuations of context
        while True:

            try:
                raw_text = input("Model input >>> ")
                while not raw_text:
                    print('Input should not be empty!')
                    raw_text = input("Model input >>> ")
            except EOFError:
                print("EOF entered, quitting.")
                exit(0)

            context_tokens = proc.encode(raw_text)
            context = torch.tensor(
                [context_tokens for _ in range(batch_size)],
                device=device)
            context_length = torch.tensor(
                [len(context_tokens) for _ in range(batch_size)],
                device=device)

            start_tokens = context[:, 0]

            helper = _get_helper(start_tokens)

            generated = 0
            for _ in range(nsamples // batch_size):
                output, _ = model(
                    context=context,
                    context_sequence_length=context_length,
                    max_decoding_length=max_decoding_length,
                    helper=helper)

                sample_id = output.sample_id
                for i in range(batch_size):
                    generated += 1
                    print("=" * 40 +
                          " SAMPLE " + str(generated) + " " + "=" * 40)
                    si = sample_id[i][len(context_tokens):]
                    print(proc.decode(si.tolist()))

            print("=" * 80)
    else:
        # Generate samples from scratch
        start_tokens = torch.full(
            [batch_size], end_token, dtype=torch.int64, device=device)

        generated = 0
        while nsamples == 0 or generated < nsamples:

            helper = _get_helper(start_tokens)

            output, _ = model(
                max_decoding_length=max_decoding_length,
                helper=helper)
            sample_id = output.sample_id
            for i in range(batch_size):
                generated += batch_size
                text = proc.decode(sample_id[i].tolist())
                print("=" * 40 +
                      " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)


if __name__ == "__main__":
    main()
