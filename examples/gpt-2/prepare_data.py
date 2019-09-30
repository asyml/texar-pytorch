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
"""Preprocesses raw data and produces pickle files
"""

import argparse
import importlib
from typing import Any

import texar.torch as tx

from utils import data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir', type=str, default='data/toy',
    help="The directory of raw data, wherein data files must be named as "
         "'train.txt', 'dev.txt', or 'test.txt'.")
parser.add_argument(
    '--max-seq-length', type=int, default=128,
    help="The maxium length of sequence, longer sequence will be trimmed.")
parser.add_argument(
    '--output-dir', type=str, default=None,
    help="The output directory where the pickle files will be generated. "
         "By default it is set to be the same as `--data-dir`.")
parser.add_argument(
    "--pretrained-model-name", type=str, default="gpt2-small",
    choices=tx.modules.GPT2Decoder.available_checkpoints(),
    help="Name of the pre-trained checkpoint to load.")
parser.add_argument(
    '--config-train', type=str, default="config_train",
    help="Configurations of GPT-2 training, including data and "
         "optimization hyperparameters.")

args = parser.parse_args()


def main() -> None:
    """Preprocess raw data and produces pickled files."""
    data_dir = args.data_dir
    if args.output_dir is None:
        pickle_output_dir = data_dir
    else:
        pickle_output_dir = args.output_dir

    tx.utils.maybe_create_dir(pickle_output_dir)

    # Create a GPT-2 tokenizer (BPE encoding)
    tokenizer = tx.data.GPT2Tokenizer(
        pretrained_model_name=args.pretrained_model_name)

    config_train: Any = importlib.import_module(args.config_train)

    # Produces pickle files
    data_utils.prepare_pickle_data(
        data_dir=data_dir,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        output_dir=pickle_output_dir,
        feature_types=config_train.feature_types)


if __name__ == "__main__":
    main()
