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
import argparse
import random

import numpy as np
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
    '--seed', type=int, default=None, help="Random seed.")
parser.add_argument(
    '--nsamples', type=int, default=1, help="The number of samples per input.")
parser.add_argument(
    '--batch-size', type=int, default=1, help="The batch size of input.")
parser.add_argument(
    '--max-decoding-length', type=int, default=128,
    help="The maximun length of generated text.")
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
    '--interactive', action='store_true', help="Interactive mode or not.")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    nsamples = args.nsamples
    batch_size = args.batch_size
    max_decoding_length = args.max_decoding_length

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
    end_token = tokenizer.map_token_to_id('<|endoftext|>')

    print("\nFinished loading\n")

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

    if args.interactive:
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

            context_tokens = tokenizer.map_text_to_id(raw_text)
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
                    print(tokenizer.map_id_to_text(si.tolist()))

            print("=" * 80)
    else:
        # Generate samples from scratch
        start_tokens = torch.full(
            (batch_size,), end_token, dtype=torch.int64, device=device)

        generated = 0
        while nsamples == 0 or generated < nsamples:

            helper = _get_helper(start_tokens)

            output, _ = model(
                max_decoding_length=max_decoding_length,
                helper=helper)
            sample_id = output.sample_id
            for i in range(batch_size):
                generated += batch_size
                text = tokenizer.map_id_to_text(sample_id[i].tolist())
                print("=" * 40 +
                      " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)


if __name__ == "__main__":
    main()
