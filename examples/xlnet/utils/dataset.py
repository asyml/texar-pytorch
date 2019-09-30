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
Dataset construction routines. Adapted from
https://github.com/zihangdai/xlnet/blob/master/run_classifier.py#L395-L508
"""

import collections
import logging
import os
from typing import Any, Dict, Optional

import torch
import tqdm
import texar.torch as tx

from utils import data_utils
from utils.processor import DataProcessor, get_processor_class

__all__ = [
    "construct_dataset",
    "load_datasets",
]


def get_record_feature_types(seq_length: int, is_regression: bool):
    name_to_features = {
        "input_ids": (torch.long, 'stacked_tensor', seq_length),
        "input_mask": (torch.float, 'stacked_tensor', seq_length),
        "segment_ids": (torch.long, 'stacked_tensor', seq_length),
        "label_ids": (torch.long, 'stacked_tensor'),
        "is_real_example": (torch.long, 'stacked_tensor'),
    }
    if is_regression:
        name_to_features["label_ids"] = (torch.float, 'stacked_tensor')
    return name_to_features


def construct_dataset(processor: DataProcessor, output_dir: str,
                      max_seq_length: int, tokenizer: tx.data.XLNetTokenizer,
                      file_prefix: Optional[str] = None,
                      overwrite_data: bool = False):
    """Convert a set of `InputExample`s to a TFRecord file."""

    file_prefix = '' if file_prefix is None else file_prefix + '.'
    # do not create duplicated records
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif (os.path.exists(os.path.join(output_dir, f"{file_prefix}train.pkl"))
          and not overwrite_data):
        logging.info("Processed dataset with prefix \"%s\" exists in %s, will "
                     "not overwrite.", file_prefix, output_dir)
        return

    logging.info("Creating dataset in directory %s.", output_dir)

    feature_types = get_record_feature_types(
        max_seq_length, is_regression=processor.is_regression)

    split_examples = {
        'train': processor.get_train_examples(),
        'dev': processor.get_dev_examples(),
    }
    try:
        split_examples['test'] = processor.get_test_examples()
    except (TypeError, NotImplementedError):
        pass

    for split, examples in split_examples.items():
        output_file = os.path.join(output_dir, f"{file_prefix}{split}.pkl")
        writer = tx.data.RecordData.writer(output_file, feature_types)
        for example in tqdm.tqdm(examples, ncols=80):
            feature = data_utils.convert_single_example(
                example, processor.labels, max_seq_length, tokenizer)

            features: Dict[str, Any] = collections.OrderedDict()
            features["input_ids"] = feature.input_ids
            features["input_mask"] = feature.input_mask
            features["segment_ids"] = feature.segment_ids
            if not processor.is_regression:
                features["label_ids"] = feature.label_id
            else:
                features["label_ids"] = float(feature.label_id)
            features["is_real_example"] = int(feature.is_real_example)

            writer.write(features)
        writer.close()


def load_datasets(task: str, input_dir: str, seq_length: int, batch_size: int,
                  drop_remainder: bool = False,
                  file_prefix: Optional[str] = None,
                  eval_batch_size: Optional[int] = None,
                  shuffle_buffer: Optional[int] = None,
                  device: Optional[torch.device] = None) \
        -> Dict[str, tx.data.RecordData]:
    r"""Creates an `input_fn` closure to be passed to TPUEstimator."""
    processor_class = get_processor_class(task)
    file_prefix = '' if file_prefix is None else file_prefix + '.'
    eval_batch_size = eval_batch_size or batch_size

    feature_types = get_record_feature_types(
        seq_length, processor_class.is_regression)

    logging.info("Loading records with prefix \"%s\" from %s",
                 file_prefix, input_dir)

    datasets = {}
    for split in ['train', 'dev', 'test']:
        is_training = (split == 'train')
        input_file = os.path.join(input_dir, f"{file_prefix}{split}.pkl")
        if not os.path.exists(input_file):
            logging.warning("%s set does not exist for task %s",
                            split.capitalize(), processor_class.task_name)
            continue
        datasets[split] = tx.data.RecordData(
            hparams={
                "dataset": {
                    "files": [input_file],
                    "feature_types": feature_types,
                },
                "batch_size": (batch_size if is_training else eval_batch_size),
                "allow_smaller_final_batch": not drop_remainder,
                # "shuffle": is_training,
                "shuffle": True,
                "shuffle_buffer_size": shuffle_buffer,
            }).to(device)

    return datasets
