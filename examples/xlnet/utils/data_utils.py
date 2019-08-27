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

from typing import Callable, Iterator, List, TypeVar

import texar.torch as tx

__all__ = [
    "repeat",
    "convert_single_example",
]

T = TypeVar('T')


def repeat(fn: Callable[[], Iterator[T]]) -> Iterator[T]:
    while True:
        yield from fn()


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


def _truncate_seq_pair(tokens_a: List[int], tokens_b: List[int],
                       max_length: int):
    r"""Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then each
    # token that's truncated likely contains more information than a longer
    # sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


VOCAB_SIZE = 32000

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


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

    tokens_a = tokenizer.map_text_to_id(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.map_text_to_id(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for two [SEP] & one [CLS] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]

    tokens = tokens_a + [tokenizer.map_token_to_id(tokenizer.sep_token)]
    segment_ids = [SEG_ID_A] * (len(tokens_a) + 1)

    if tokens_b:
        tokens += tokens_b + [tokenizer.map_token_to_id(tokenizer.sep_token)]
        segment_ids += [SEG_ID_B] * (len(tokens_b) + 1)

    tokens.append(tokenizer.map_token_to_id(tokenizer.cls_token))
    segment_ids.append(SEG_ID_CLS)

    input_ids = tokens

    # The mask has 0 for real tokens and 1 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < max_seq_length:
        delta_len = max_seq_length - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

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
