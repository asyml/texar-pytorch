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

from typing import Any, Dict

from texar.torch.data.tokenizers.pretrained_gpt2_tokenizer import GPT2Tokenizer

__all__ = [
    'RoBERTaTokenizer',
]

_GPT2_PATH = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/"
_CHECKPOINT_FILES = ["encoder.json", "vocab.bpe"]


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
            a default directory (user's home directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    _MODEL2URL = {
        'roberta-base': [_GPT2_PATH + f"{file}" for file in _CHECKPOINT_FILES],
        'roberta-large': [_GPT2_PATH + f"{file}" for file in _CHECKPOINT_FILES],
    }
    _MAX_INPUT_SIZE = {
        'roberta-base': 512,
        'roberta-large': 512,
    }

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
            'cls_token': '</s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>',
            'errors': 'replace',
        }
