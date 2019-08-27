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
Pre-trained BERT tokenizer.

Code structure adapted from:
    `https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_bert.py`
"""

from typing import Any, Dict, List, Optional, Tuple

import os

from texar.torch.modules.pretrained.pretrained_bert import PretrainedBERTMixin
from texar.torch.data.tokenizers.pretrained_tokenizer_base import \
    PretrainedTokenizerBase
from texar.torch.data.tokenizers.pretrained_bert_tokenizer_utils import \
    load_vocab, BasicTokenizer, WordpieceTokenizer

__all__ = [
    'BERTTokenizer',
]


class BERTTokenizer(PretrainedBERTMixin, PretrainedTokenizerBase):
    r"""Pre-trained BERT Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `bert-base-uncased`). Please refer to
            :class:`~texar.torch.modules.PretrainedBERTMixin` for
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

    _MAX_INPUT_SIZE = {
        'bert-base-uncased': 512,
        'bert-large-uncased': 512,
        'bert-base-cased': 512,
        'bert-large-cased': 512,
        'bert-base-multilingual-uncased': 512,
        'bert-base-multilingual-cased': 512,
        'bert-base-chinese': 512,
    }
    _VOCAB_FILE_NAMES = {'vocab_file': 'vocab.txt'}

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        self.load_pretrained_config(pretrained_model_name, cache_dir, hparams)

        super().__init__(hparams=None)

        self.config = {
            'tokenize_chinese_chars': self.hparams['tokenize_chinese_chars'],
            'do_lower_case': self.hparams['do_lower_case'],
            'do_basic_tokenize': self.hparams['do_basic_tokenize'],
            'non_split_tokens': self.hparams['non_split_tokens'],
        }

        if self.pretrained_model_dir is not None:
            vocab_file = os.path.join(self.pretrained_model_dir,
                                      self._VOCAB_FILE_NAMES['vocab_file'])
            assert self.pretrained_model_name is not None
            if self._MAX_INPUT_SIZE.get(self.pretrained_model_name):
                self.max_len = self._MAX_INPUT_SIZE[self.pretrained_model_name]
        else:
            vocab_file = self.hparams['vocab_file']
            if self.hparams.get('max_len'):
                self.max_len = self.hparams['max_len']

        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path "
                             "'{}".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = dict((ids, tok) for tok, ids in self.vocab.items())

        self.do_basic_tokenize = self.hparams['do_basic_tokenize']
        if self.do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=self.hparams["do_lower_case"],
                never_split=self.hparams["non_split_tokens"],
                tokenize_chinese_chars=self.hparams["tokenize_chinese_chars"])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                      unk_token=self.unk_token)

    def _map_text_to_token(self, text: str) -> List[str]:  # type: ignore
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                    text, never_split=self.all_special_tokens):
                assert token is not None
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def save_vocab(self, save_directory: str) -> Tuple[str]:
        r"""Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(save_directory):
            save_directory = os.path.join(save_directory,
                                          self._VOCAB_FILE_NAMES['vocab_file'])
        with open(save_directory, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(),
                                             key=lambda kv: kv[1]):
                if index != token_index:
                    print("Saving vocabulary to {}: vocabulary indices are not "
                          "consecutive. Please check that the vocabulary is "
                          "not corrupted!".format(save_directory))
                    index = token_index
                writer.write(token + u'\n')
                index += 1

        return (save_directory, )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _map_token_to_id(self, token: str) -> int:
        r"""Maps a token to an id using the vocabulary."""
        unk_id = self.vocab.get(self.unk_token)
        assert unk_id is not None
        return self.vocab.get(token, unk_id)

    def _map_id_to_token(self, index: int) -> str:
        r"""Maps an id to a token using the vocabulary.
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def map_token_to_text(self, tokens: List[str]) -> str:
        r"""Maps a sequence of tokens (string) to a single string."""
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

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
                "pretrained_model_name": "bert-base-uncased",
                "vocab_file": None,
                "max_len": 512,
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "tokenize_chinese_chars": True,
                "do_lower_case": True,
                "do_basic_tokenize": True,
                "non_split_tokens": None,
                "name": "bert_tokenizer",
            }

        Here:

        `"pretrained_model_name"`: str or None
            The name of the pre-trained BERT model.

        `"vocab_file"`: str or None
            The path to a one-wordpiece-per-line vocabulary file.

        `"max_len"`: int
            The maximum sequence length that this model might ever be used with.

        `"unk_token"`: str
            Unknown token.

        `"sep_token"`: str
            Separation token.

        `"pad_token"`: str
            Padding token.

        `"cls_token"`: str
            Classification token.

        `"mask_token"`: str
            Masking token.

        `"tokenize_chinese_chars"`: bool
            Whether to tokenize Chinese characters.

        `"do_lower_case"`: bool
            Whether to lower case the input
            Only has an effect when `do_basic_tokenize=True`

        `"do_basic_tokenize"`: bool
            Whether to do basic tokenization before wordpiece.

        `"non_split_tokens"`: list
            List of tokens which will never be split during tokenization.
            Only has an effect when `do_basic_tokenize=True`

        `"name"`: str
            Name of the tokenizer.

        """
        return {
            'pretrained_model_name': 'bert-base-uncased',
            'vocab_file': None,
            'max_len': 512,
            'unk_token': '[UNK]',
            'sep_token': '[SEP]',
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'mask_token': '[MASK]',
            'tokenize_chinese_chars': True,
            'do_lower_case': True,
            'do_basic_tokenize': True,
            'non_split_tokens': None,
            'name': 'bert_tokenizer',
            '@no_typecheck': ['pretrained_model_name'],
        }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str):
        r"""Returns the configuration of the pre-trained BERT tokenizer."""
        return {
            'vocab_file': None,
            'max_len': 512,
            'unk_token': '[UNK]',
            'sep_token': '[SEP]',
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'mask_token': '[MASK]',
            'tokenize_chinese_chars': True,
            'do_lower_case': True,
            'do_basic_tokenize': True,
            'non_split_tokens': None,
        }
