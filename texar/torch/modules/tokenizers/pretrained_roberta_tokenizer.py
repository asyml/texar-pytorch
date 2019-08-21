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

from texar.torch.modules.tokenizers.pretrained_gpt2_tokenizer import \
    GPT2Tokenizer

__all__ = [
    'RoBERTaTokenizer',
]

_GPT2_PATH = "https://storage.googleapis.com/gpt-2/models/"
_CHECKPOINT_FILES = ["encoder.json", "vocab.bpe"]


class RoBERTaTokenizer(GPT2Tokenizer):
    r"""Pre-trained RoBERTa Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `roberta-base`). Please refer to
            :class:`~texar.torch.modules.pretrained.PretrainedRoBERTaMixin` for
            all supported models.
            If None, the model name in :attr:hparams is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    _MODEL2URL = {
        'roberta-base': [_GPT2_PATH + f"117M/{file}"
                         for file in _CHECKPOINT_FILES],
        'roberta-large': [_GPT2_PATH + f"117M/{file}"
                          for file in _CHECKPOINT_FILES],
    }
    _MAX_INPUT_SIZE = {
        'roberta-base': 512,
        'roberta-large': 512,
    }

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
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
