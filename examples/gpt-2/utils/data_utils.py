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
"""
Utils of data preprocessing for GPT2 training.
"""

from typing import Any, Dict, List, Optional

import os

import texar.torch as tx


def process_single_text(raw_text: str,
                        max_seq_length: int,
                        encoder: tx.data.GPT2Tokenizer,
                        use_eos_token: Optional[bool]):
    """Processes a single piece of text. Performs BPE encoding,
    converting to indexes, truncation, and padding, etc.
    """
    input_ids, input_mask = encoder.add_special_tokens_single_sequence(
        text=raw_text, max_length=max_seq_length, use_eos_token=use_eos_token)
    token_length = sum(input_mask)

    return input_ids, token_length


def read_raw_data(data_fn: str):
    r"""
    Reads raw data from a file. Each line contains one example.
    """
    examples = []
    with open(data_fn, "r") as fin:
        for line in fin:
            examples.append(line.strip())
    return examples


def file_based_convert_examples_to_features(
        examples: List[str],
        max_seq_length: int,
        encoder: tx.data.GPT2Tokenizer,
        output_file: str,
        feature_original_types: Dict[str, Any],
        use_eos_token: Optional[bool] = True):
    r"""Converts a set of examples to a `pickle` file."""

    with tx.data.RecordData.writer(
            output_file, feature_original_types) as writer:

        for (_, example) in enumerate(examples):

            text_ids, length = process_single_text(
                example, max_seq_length, encoder, use_eos_token)

            features = {
                "text_ids": text_ids,
                "length": length
            }
            writer.write(features)  # type: ignore


def prepare_pickle_data(data_dir: str,
                        max_seq_length: int,
                        encoder: tx.data.GPT2Tokenizer,
                        output_dir: str,
                        feature_original_types: Dict[str, Any]):
    r"""Prepare the `pickle` dataset.
    Args:
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        encoder: The GPT-2 tokenizer.
        output_dir: The directory to save the pickled files in.
        feature_original_types: The original type of the feature.
    """
    train_fn = os.path.join(data_dir, "train.txt")
    if os.path.isfile(train_fn):
        print("Processing %s" % train_fn)
        train_examples = read_raw_data(train_fn)
        train_file = os.path.join(output_dir, "train.pkl")
        file_based_convert_examples_to_features(
            train_examples, max_seq_length, encoder, train_file,
            feature_original_types)

    dev_fn = os.path.join(data_dir, "dev.txt")
    if os.path.isfile(dev_fn):
        print("Processing %s" % dev_fn)
        eval_examples = read_raw_data(dev_fn)
        eval_file = os.path.join(output_dir, "dev.pkl")
        file_based_convert_examples_to_features(
            eval_examples, max_seq_length, encoder, eval_file,
            feature_original_types)

    test_fn = os.path.join(data_dir, "test.txt")
    if os.path.isfile(test_fn):
        print("Processing %s" % test_fn)
        test_examples = read_raw_data(test_fn)
        test_file = os.path.join(output_dir, "test.pkl")
        file_based_convert_examples_to_features(
            test_examples, max_seq_length, encoder, test_file,
            feature_original_types, use_eos_token=False)
