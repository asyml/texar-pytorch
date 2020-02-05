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
Pre-trained RoBERTa tokenizer.
"""

from typing import Any, Dict, List, Optional, Tuple

from texar.torch.data.tokenizers.gpt2_tokenizer import GPT2Tokenizer
from texar.torch.utils.utils import truncate_seq_pair

__all__ = [
    'RoBERTaTokenizer',
]

_ROBERTA_PATH = "https://s3.amazonaws.com/models.huggingface.co/bert/"


class RoBERTaTokenizer(GPT2Tokenizer):
    r"""Pre-trained RoBERTa Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `roberta-base`). Please refer to
            :class:`~texar.torch.modules.PretrainedRoBERTaMixin` for
            all supported models.
            If None, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    _MODEL2URL = {
        'roberta-base': [
            _ROBERTA_PATH + 'roberta-base-vocab.json',
            _ROBERTA_PATH + 'roberta-base-merges.txt',
        ],
        'roberta-large': [
            _ROBERTA_PATH + 'roberta-large-vocab.json',
            _ROBERTA_PATH + 'roberta-large-merges.txt',
        ],
    }
    _MAX_INPUT_SIZE = {
        'roberta-base': 512,
        'roberta-large': 512,
    }
    _VOCAB_FILE_MAP = {
        'vocab_file': {
            'roberta-base': 'roberta-base-vocab.json',
            'roberta-large': 'roberta-large-vocab.json',
        },
        'merges_file': {
            'roberta-base': 'roberta-base-merges.txt',
            'roberta-large': 'roberta-large-merges.txt',
        },
    }

    def encode_text(self,  # type: ignore
                    text_a: str,
                    text_b: Optional[str] = None,
                    max_seq_length: Optional[int] = None) -> \
            Tuple[List[int], List[int]]:
        r"""Adds special tokens to a sequence or sequence pair and computes the
        corresponding input mask for RoBERTa specific tasks.
        The sequence will be truncated if its length is larger than
        ``max_seq_length``.

        A RoBERTa sequence has the following format:
        `[cls_token]` X `[sep_token]`

        A RoBERTa sequence pair has the following format:
        `[cls_token]` A `[spe_token]` `[sep_token]` B `[sep_token]`

        Args:
            text_a: The first input text.
            text_b: The second input text.
            max_seq_length: Maximum sequence length.

        Returns:
            A tuple of `(input_ids, segment_ids, input_mask)`, where

            - ``input_ids``: A list of input token ids with added
              special token ids.
            - ``input_mask``: A list of mask ids. The mask has 1 for real
              tokens and 0 for padding tokens. Only real tokens are
              attended to.
        """
        if max_seq_length is None:
            max_seq_length = self.max_len

        cls_token_id = self._map_token_to_id(self.cls_token)
        sep_token_id = self._map_token_to_id(self.sep_token)

        token_ids_a = self.map_text_to_id(text_a)
        assert isinstance(token_ids_a, list)

        token_ids_b = None
        if text_b:
            token_ids_b = self.map_text_to_id(text_b)

        if token_ids_b:
            assert isinstance(token_ids_b, list)
            # Modifies `token_ids_a` and `token_ids_b` in place so that the
            # total length is less than the specified length.
            # Account for <s>, </s>, </s>, </s> with "- 4"
            truncate_seq_pair(token_ids_a, token_ids_b, max_seq_length - 4)

            input_ids = ([cls_token_id] + token_ids_a + [sep_token_id] +
                         [sep_token_id] + token_ids_b + [sep_token_id])
        else:
            # Account for <s> and </s> with "- 2"
            token_ids_a = token_ids_a[:max_seq_length - 2]

            input_ids = [cls_token_id] + token_ids_a + [sep_token_id]

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the maximum sequence length.
        input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
        input_mask = input_mask + [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, input_mask

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        * The tokenizer is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the tokenizer is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the tokenizer is defined by the
          configurations in `hparams`.

        .. code-block:: python

            {
                "pretrained_model_name": "roberta-base",
                "vocab_file": None,
                "merges_file": None,
                "max_len": 512,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "sep_token": "</s>",
                "cls_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>",
                "errors": "replace",
                "name": "roberta_tokenizer",
            }

        Here:

        `"pretrained_model_name"`: str or None
            The name of the pre-trained RoBERTa model.

        `"vocab_file"`: str or None
            The path to a vocabulary json file mapping tokens to ids.

        `"merges_file"`: str or None
            The path to a merges file.

        `"max_len"`: int
            The maximum sequence length that this model might ever be used with.

        `"bos_token"`: str
            Beginning of sentence token.

        `"eos_token"`: str
            End of sentence token.

        `"sep_token"`: str
            Separation token.

        `"cls_token"`: str
            Classification token.

        `"unk_token"`: str
            Unknown token.

        `"pad_token"`: str
            Padding token.

        `"mask_token"`: str
            Masking token.

        `"errors"`: str
            Response when decoding fails. The possible values are
            `ignore`, `replace`, and `strict`.

        `"name"`: str
            Name of the tokenizer.
        """
        return {
            'pretrained_model_name': 'roberta-base',
            'vocab_file': None,
            'merges_file': None,
            'max_len': 512,
            'bos_token': '<s>',
            'eos_token': '</s>',
            'sep_token': '</s>',
            'cls_token': '</s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>',
            'errors': 'replace',
            'name': 'roberta_tokenizer',
            '@no_typecheck': ['pretrained_model_name'],
        }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str):
        r"""Returns the configuration of the pre-trained RoBERTa tokenizer."""
        return {
            'vocab_file': None,
            'merges_file': None,
            'max_len': 512,
            'bos_token': '<s>',
            'eos_token': '</s>',
            'sep_token': '</s>',
            'cls_token': '<s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>',
            'errors': 'replace',
        }
