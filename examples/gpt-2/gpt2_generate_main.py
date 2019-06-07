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
import importlib
import random

import numpy as np
import torch
from torch import nn

import texar as tx
from texar.modules.embedders.embedders import WordEmbedder
from texar.modules.embedders.position_embedders import PositionEmbedder
from texar.modules.decoders.transformer_decoders import TransformerDecoder
from utils import model_utils, processor

# pylint: disable=invalid-name, too-many-locals, too-many-statements, no-member
# pylint: disable=too-many-branches

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',
                    type=str,
                    default=None,
                    help="Model checkpoint to load model weights from. Use "
                         "`--pretrain_checkpoint` instead if loading OpenAI "
                         "pretrained checkpoint.")
parser.add_argument('--pretrain_checkpoint',
                    type=str,
                    default="gpt2_pretrained_models/model_117M/model.ckpt",
                    help="OpenAI pretrained model checkpoint. Ignored if "
                         "'--checkpoint' is specified.")
parser.add_argument('--pretrain_model_dir',
                    type=str,
                    default="gpt2_pretrained_models/model_117M",
                    help="The directory of pretrained model, for loading "
                         "vocabuary, etc.")
parser.add_argument('--seed',
                    type=int,
                    default=None,
                    help="Random seed.")
parser.add_argument('--nsamples',
                    type=int,
                    default=1,
                    help="The number of samples per input.")
parser.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help="The batch size of input.")
parser.add_argument('--max_decoding_length',
                    type=int,
                    default=128,
                    help="The maximun length of generated text.")
parser.add_argument('--temperature',
                    type=float,
                    default=0.7,
                    help="Softmax temperature for top-k sample decoding. "
                         "Must be strictly greater than 0. Defaults to 0.7.")
parser.add_argument('--top_k',
                    type=int,
                    default=40,
                    help="The number of top most likely candidates from a "
                         "vocab distribution.")
parser.add_argument('--is_interactive',
                    action='store_true',
                    help="Interactive mode or not.")
parser.add_argument('--config_type',
                    type=str,
                    default="texar",
                    help="The configuration file type. Set to 'json' if the "
                         "GPT-2 config file is in the same type of the "
                         "official GPT-2 config file. Set to 'texar' "
                         "if GPT-2 config file is in Texar type.")
parser.add_argument('--config_model',
                    type=str,
                    default="configs.config_model_117M",
                    help="The model configuration file to configure the "
                         "model. The config file type is define by the "
                         "'config_type',it be of texar type or json type."
                         "For '--config_type=json', set the json "
                         "config file path like: '--config_model "
                         "gpt2_pretrained_models/model_117M/hparams.json';"
                         "For '--config_type=texar', set the texar "
                         "config file like: "
                         "'--config_model configs.config_model_117M'.")

args = parser.parse_args()


class GPT2(nn.Module):
    def __init__(self, gpt2_config, top_k, temperature):
        super().__init__()
        self.word_embedder = WordEmbedder(
            vocab_size=gpt2_config.vocab_size,
            hparams=gpt2_config.embed)

        self.pos_embedder = PositionEmbedder(
            position_size=gpt2_config.position_size,
            hparams=gpt2_config.pos_embed)

        self.decoder = TransformerDecoder(
            vocab_size=gpt2_config.vocab_size,
            output_layer=self.word_embedder.embedding,
            hparams=gpt2_config.decoder)

        self.top_k = top_k
        self.temperature = temperature

        self._embedding_fn = lambda x, y: (
                self.word_embedder(x) + self.pos_embedder(y))

    def forward(self, start_tokens, end_token, context, context_sequence_length,
                max_decoding_length):
        helper = tx.modules.TopKSampleEmbeddingHelper(
            embedding=self._embedding_fn,
            start_tokens=start_tokens,
            end_token=end_token,
            top_k=self.top_k,
            softmax_temperature=self.temperature)
        output, _ = self.decoder(
            context=context,
            context_sequence_length=context_sequence_length,
            max_decoding_length=max_decoding_length,
            helper=helper)
        return output


def run_model():
    r"""Build the model and run.
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

    # Load GPT-2 model configuration
    if args.config_type == "json":
        gpt2_config = model_utils.transform_gpt2_to_texar_config(
            args.config_model)
    elif args.config_type == 'texar':
        gpt2_config = importlib.import_module(
            args.config_model)
    else:
        raise ValueError('Unknown config_type.')

    assert max_decoding_length <= gpt2_config.position_size, (
        "max_decoding_length should not be greater than position size")
    assert nsamples % batch_size == 0, (
        "nsamples must be dividable by batch_size")

    # Create a data pre-processor for, e.g., BPE encoding
    proc = processor.get_encoder(args.pretrain_model_dir)
    end_token = proc.encoder['<|endoftext|>']

    # Build the GPT-2 model
    model = GPT2(gpt2_config, args.top_k, args.temperature)

    # Load model checkpoint
    if args.checkpoint:
        model_utils.init_gpt2_checkpoint(model, args.checkpoint)
    elif args.pretrain_checkpoint:
        model_utils.init_gpt2_checkpoint(model, args.pretrain_checkpoint)

    if torch.cuda.is_available():
        model.cuda()
        device = torch.cuda.current_device()
    else:
        device = None

    print("\nFinished loading\n")

    if args.is_interactive:
        # Generate continuations of context
        while True:

            try:
                raw_text = input("Model input >>> ")
                while not raw_text:
                    print('Input should not be empty!')
                    raw_text = input("Model input >>> ")
            except EOFError:
                exit(0)

            context_tokens = proc.encode(raw_text)
            context = torch.tensor(
                [context_tokens for _ in range(batch_size)], device=device)
            context_length = torch.tensor(
                [len(context_tokens) for _ in range(batch_size)], device=device)

            start_tokens = context[:, 0]

            generated = 0
            for _ in range(nsamples // batch_size):
                output = model(
                    start_tokens=start_tokens,
                    end_token=end_token,
                    context=context,
                    context_sequence_length=context_length,
                    max_decoding_length=max_decoding_length)

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

            output = model(
                start_tokens=start_tokens,
                end_token=end_token,
                max_decoding_length=max_decoding_length)
            sample_id = output.sample_id
            for i in range(batch_size):
                generated += batch_size
                text = proc.decode(sample_id[i].tolist())
                print("=" * 40 +
                      " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)


if __name__ == '__main__':
    run_model()
