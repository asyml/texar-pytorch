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
Utils of pre-trained ELMo tokenizer.

Code adapted from:
    `https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/elmo_indexer.py`
"""
from typing import Dict, List, Optional

import torch

from torch.nn.utils.rnn import pad_sequence


__all__ = [
    "ELMoCharacterMapper",
    "batch_to_ids",
]


def _make_bos_eos(
    character: int,
    padding_character: int,
    beginning_of_word_character: int,
    end_of_word_character: int,
    max_word_length: int,
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoCharacterMapper:
    r"""Maps individual tokens to sequences of character ids, compatible with
    ELMo. To be consistent with previously trained models, we include it here as
    special of existing character indexers.

    We allow to add optional additional special tokens with designated
    character ids with `tokens_to_add`.
    """

    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260  # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )

    bos_token = "<S>"
    eos_token = "</S>"

    def __init__(self, tokens_to_add: Optional[Dict[str, int]] = None) -> None:
        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str) -> List[int]:
        if word in self.tokens_to_add:
            char_ids = ([ELMoCharacterMapper.padding_character] *
                        ELMoCharacterMapper.max_word_length)
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = ELMoCharacterMapper.end_of_word_character
        elif word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode(
                "utf-8", "ignore")[: (ELMoCharacterMapper.max_word_length - 2)]
            char_ids = ([ELMoCharacterMapper.padding_character] *
                        ELMoCharacterMapper.max_word_length)
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = \
                ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
    r"""Converts a batch of tokenized sentences to a tensor representing the
    sentences with encoded characters (len(batch), max sentence length,
    max word length).

    Args:
        batch: A list of tokenized sentences.

    Returns:
        A tensor of padded character ids.
    """
    res = []
    mapper = ELMoCharacterMapper()
    for sentence in batch:
        character_ids = [mapper.convert_word_to_char_ids(token)
                         for token in sentence]
        res.append(torch.tensor(character_ids))

    return pad_sequence(res, batch_first=True)
