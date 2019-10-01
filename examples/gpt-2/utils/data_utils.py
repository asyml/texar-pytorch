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

from typing import Any, Dict, List

import os

import texar.torch as tx


def read_raw_data(data_fn: str):
    r"""
    Reads raw data from a file. Each line contains one example.
    """
    examples = []
    with open(data_fn, "r") as fin:
        for line in fin:
            examples.append(line.strip())
    return examples


def convert_examples_to_features_and_output_to_files(
        examples: List[str],
        max_seq_length: int,
        tokenizer: tx.data.GPT2Tokenizer,
        output_file: str,
        feature_types: Dict[str, Any],
        append_eos_token: bool = True):
    r"""Converts a set of examples to a `pickle` file."""

    with tx.data.RecordData.writer(output_file, feature_types) as writer:

        for (_, example) in enumerate(examples):

            text_ids, length = tokenizer.encode_text(
                text=example, max_seq_length=max_seq_length,
                append_eos_token=append_eos_token)

            features = {
                "text_ids": text_ids,
                "length": length
            }
            writer.write(features)  # type: ignore


def prepare_pickle_data(data_dir: str,
                        max_seq_length: int,
                        tokenizer: tx.data.GPT2Tokenizer,
                        output_dir: str,
                        feature_types: Dict[str, Any]):
    r"""Prepare the `pickle` dataset.
    Args:
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        tokenizer: The GPT-2 tokenizer.
        output_dir: The directory to save the pickled files in.
        feature_types: The original type of the feature.
    """
    train_fn = os.path.join(data_dir, "train.txt")
    if os.path.isfile(train_fn):
        print("Processing %s" % train_fn)
        train_examples = read_raw_data(train_fn)
        train_file = os.path.join(output_dir, "train.pkl")
        convert_examples_to_features_and_output_to_files(
            train_examples, max_seq_length, tokenizer, train_file,
            feature_types)

    dev_fn = os.path.join(data_dir, "dev.txt")
    if os.path.isfile(dev_fn):
        print("Processing %s" % dev_fn)
        eval_examples = read_raw_data(dev_fn)
        eval_file = os.path.join(output_dir, "dev.pkl")
        convert_examples_to_features_and_output_to_files(
            eval_examples, max_seq_length, tokenizer, eval_file,
            feature_types)

    test_fn = os.path.join(data_dir, "test.txt")
    if os.path.isfile(test_fn):
        print("Processing %s" % test_fn)
        test_examples = read_raw_data(test_fn)
        test_file = os.path.join(output_dir, "test.pkl")
        convert_examples_to_features_and_output_to_files(
            test_examples, max_seq_length, tokenizer, test_file,
            feature_types, append_eos_token=False)
