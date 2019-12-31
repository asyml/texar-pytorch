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
Pre-trained T5 tokenizer.
"""

from typing import Any, Dict, Optional

import os
import re

from texar.torch.data.tokenizers.sentencepiece_tokenizer \
    import SentencePieceTokenizer
from texar.torch.modules.pretrained.t5 import PretrainedT5Mixin

__all__ = [
    'T5Tokenizer',
]


class T5Tokenizer(SentencePieceTokenizer, PretrainedT5Mixin):
    r"""Pre-trained T5 Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `T5-Small`). Please refer to
            :class:`~texar.torch.modules.PretrainedT5Mixin` for
            all supported models.
            If None, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    _IS_PRETRAINED = True

    _VOCAB_FILE_NAMES = {
        'vocab_file': 'sentencepiece.model'
    }

    _MAX_INPUT_SIZE = {
        'T5-Small': 512,
        'T5-Base': 512,
        'T5-Large': 512,
        'T5-3B': 512,
        'T5-11B': 512
    }

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):

        self.load_pretrained_config(pretrained_model_name, cache_dir, hparams)

        if self.pretrained_model_dir is not None:
            assert self.pretrained_model_name is not None
            vocab_file = os.path.join(self.pretrained_model_dir,
                                      self._VOCAB_FILE_NAMES['vocab_file'])

            if self._MAX_INPUT_SIZE.get(self.pretrained_model_name):
                self.max_len = self._MAX_INPUT_SIZE[self.pretrained_model_name]
            setattr(self.hparams, 'vocab_file', vocab_file)
        else:
            if self.hparams.get('max_len'):
                self.max_len = self.hparams['max_len']

        # Add extra_ids to the special token list
        additional_special_tokens = []
        extra_ids = self.hparams['extra_ids']
        if extra_ids > 0:
            additional_special_tokens.extend(
                ["<extra_id_{}>".format(i) for i in range(extra_ids)])

        setattr(self.hparams, 'additional_special_tokens',
                additional_special_tokens)

        super().__init__(hparams=None)

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
                "pretrained_model_name": "T5-Small",
                "vocab_file": None,
                "max_len": 512,
                "bos_token": None,
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "extra_ids": 100,
                "additional_special_tokens": [],
                "name": "t5_tokenizer",
            }

        Here:

        `"pretrained_model_name"`: str or None
            The name of the pre-trained T5 model.

        `"vocab_file"`: str or None
            The path to a sentencepiece vocabulary file.

        `"max_len"`: int or None
            The maximum sequence length that this model might ever be used with.

        `"bos_token"`: str or None
            Beginning of sentence token. Set None to disable ``bos_token``.

        `"eos_token"`: str
            End of sentence token. Set None to disable ``eos_token``.

        `"unk_token"`: str
            Unknown token. Set None to disable ``unk_token``.

        `"pad_token"`: str
            Padding token. Set None to disable ``pad_token``.

        `"extra_ids"`: int
            Add a number of extra ids added to the end of the vocabulary for
            use as sentinels. These tokens are accessible as `<extra_id_{%d}>`
            where `{%d}` is a number between 0 and extra_ids-1. Extra tokens
            are indexed from the end of the vocabulary up to beginning
            (<extra_id_0> is the last token in the vocabulary) (like in T5
            preprocessing) see:
            `https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117`

        `"additional_special_tokens"`: list
            A list of additional special tokens.

        `"name"`: str
            Name of the tokenizer.
        """
        return {
            'pretrained_model_name': 'T5-Small',
            'vocab_file': None,
            'max_len': 512,
            'bos_token': None,
            'eos_token': '</s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'extra_ids': 100,
            'additional_special_tokens': [],
            'name': 't5_tokenizer',
            '@no_typecheck': ['pretrained_model_name'],
        }

    @property
    def vocab_size(self) -> int:
        return len(self.sp_model) + self.hparams['extra_ids']

    def _map_token_to_id(self, token: str) -> int:
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))  # type: ignore
            return self.vocab_size - num - 1
        return self.sp_model.PieceToId(token)

    def _map_id_to_token(self, index: int) -> str:
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            token = "<extra_id_{}>".format(self.vocab_size - 1 - index)
        return token
