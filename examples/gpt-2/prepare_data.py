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

import texar as tx

from utils import data_utils, processor


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', type=str, default='data/toy',
    help="The directory of raw data, wherein data files must be named as "
         "'train.txt', 'dev.txt', or 'test.txt'.")
parser.add_argument(
    '--max_seq_length', type=int, default=128,
    help="The maxium length of sequence, longer sequence will be trimmed.")
parser.add_argument(
    '--output_dir', type=str, default=None,
    help="The output directory where the pickle files will be generated. "
         "By default it is set to be the same as `--data_dir`.")
parser.add_argument(
    '--pretrained_model_name', type=str, default='117M',
    help="The name of a pre-trained model to load selected in the "
         "list of: `117M`, `345M`.")

args = parser.parse_args()


def prepare_data():
    r"""Preprocesses raw data and produces pickle files.
    """
    data_dir = args.data_dir
    if args.output_dir is None:
        pickle_output_dir = data_dir
    else:
        pickle_output_dir = args.output_dir

    tx.utils.maybe_create_dir(pickle_output_dir)

    pretrained_model_dir = tx.modules.load_pretrained_gpt2(
        pretrained_model_name=args.pretrained_model_name,
        cache_dir='gpt2_pretrained_models')

    # Creates a data pre-processor for, e.g., BPE encoding
    proc = processor.get_encoder(pretrained_model_dir)

    from configs.config_train import feature_original_types

    # Produces pickle files
    data_utils.prepare_pickle_data(
        data_dir=data_dir,
        max_seq_length=args.max_seq_length,
        encoder=proc,
        output_dir=pickle_output_dir,
        feature_original_types=feature_original_types)


def main():
    """Data preparation.
    """
    prepare_data()


if __name__ == "__main__":
    main()
