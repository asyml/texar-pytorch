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
Data preprocessing utilities. Adapted from
https://github.com/zihangdai/xlnet/blob/master/classifier_utils.py
https://github.com/zihangdai/xlnet/blob/master/prepro_utils.py
"""

from typing import List, TypeVar

import texar.torch as tx

__all__ = [
    "convert_single_example",
]

T = TypeVar('T')


class PaddingInputExample:
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it
    means the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    batches could cause silent errors.
    """


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_single_example(example, label_list: List[str], max_seq_length: int,
                           tokenizer: tx.data.XLNetTokenizer) -> InputFeatures:
    r"""Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[1] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    input_ids, segment_ids, input_mask = \
        tokenizer.encode_text(text_a=example.text_a,
                              text_b=example.text_b,
                              max_seq_length=max_seq_length)

    if len(label_list) > 0:
        label_id = label_list.index(example.label)
    else:
        label_id = example.label

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature
