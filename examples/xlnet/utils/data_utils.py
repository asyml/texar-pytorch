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

import unicodedata
from typing import Callable, List, TypeVar, Union

import sentencepiece as spm

__all__ = [
    "TokenizeFn",
    "create_tokenize_fn",
    "convert_single_example",
]

T = TypeVar('T')


def preprocess_text(inputs: Union[str, bytes], lower: bool = False,
                    remove_space: bool = True,
                    keep_accents: bool = False) -> str:
    if isinstance(inputs, bytes):
        inputs = inputs.decode('utf-8')

    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


SPIECE_UNDERLINE = '▁'


def encode_pieces(sp_model, text, sample: bool = False):
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces: List[str] = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if (piece[0] != SPIECE_UNDERLINE and
                    cur_pieces[0][0] == SPIECE_UNDERLINE):
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    return new_pieces


def encode_ids(sp_model: spm.SentencePieceProcessor,
               text: str, sample: bool = False) -> List[int]:
    pieces = encode_pieces(sp_model, text, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


TokenizeFn = Callable[[str], List[int]]


def create_tokenize_fn(sp_model, uncased: bool = False) -> TokenizeFn:
    def tokenize_fn(text: Union[str, bytes]) -> List[int]:
        text = preprocess_text(text, lower=uncased)
        return encode_ids(sp_model, text)

    return tokenize_fn


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


special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

VOCAB_SIZE = 32000
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


def convert_single_example(example, label_list: List[str], max_seq_length: int,
                           tokenize_fn: TokenizeFn) -> InputFeatures:
    r"""Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[1] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    tokens_a = tokenize_fn(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenize_fn(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for two [SEP] & one [CLS] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]

    tokens = tokens_a + [SEP_ID]
    segment_ids = [SEG_ID_A] * (len(tokens_a) + 1)

    if tokens_b:
        tokens += tokens_b + [SEP_ID]
        segment_ids += [SEG_ID_B] * (len(tokens_b) + 1)

    tokens.append(CLS_ID)
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
