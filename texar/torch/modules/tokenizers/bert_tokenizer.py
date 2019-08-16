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
"""

from typing import Any, Dict, List, Optional, Tuple

import os

from texar.torch.modules.pretrained.pretrained_bert import PretrainedBERTMixin
from texar.torch.modules.tokenizers.pretrained_tokenizer_base import \
    PretrainedTokenizerBase
from texar.torch.modules.tokenizers.bert_tokenizer_utils import \
    load_vocab, BasicTokenizer, WordpieceTokenizer

__all__ = [
    'BERTTokenizer',
]


_BERT_PATH = "https://storage.googleapis.com/bert_models/"


class BERTTokenizer(PretrainedBERTMixin, PretrainedTokenizerBase):
    r"""Pre-trained BERT Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `bert-base-uncased`). Please refer to
            :class:`~texar.torch.modules.pretrained.PretrainedBERTMixin` for
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
        super().__init__(hparams=hparams)

        self.config = {
            'tokenize_chinese_chars': self.hparams['tokenize_chinese_chars'],
            'do_lower_case': self.hparams['do_lower_case'],
            'do_basic_tokenize': self.hparams['do_basic_tokenize'],
            'never_split': self.hparams['never_split'],
        }

        self.load_pretrained_config(pretrained_model_name, cache_dir)

        if self.pretrained_model_dir is not None:
            vocab_file = os.path.join(self.pretrained_model_dir,
                                      self._VOCAB_FILE_NAMES['vocab_file'])
            if self._MAX_INPUT_SIZE.get(pretrained_model_name):
                self.max_len = self._MAX_INPUT_SIZE[pretrained_model_name]
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
                never_split=self.hparams["never_split"],
                tokenize_chinese_chars=self.hparams["tokenize_chinese_chars"])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                      unk_token=self.unk_token)

    def forward(self, inputs, job_type, **kwargs):
        if job_type == 'text-to-token':
            return self.tokenize(inputs)
        elif job_type == 'token-to-text':
            return self.convert_tokens_to_string(inputs)
        elif job_type == 'text-to-id':
            return self.encode(inputs)
        elif job_type == 'id-to-text':
            skip_special_tokens = kwargs.get('skip_special_tokens', False)
            clean_up_tokenization_spaces = kwargs.get(
                'clean_up_tokenization_spaces', True)
            return self.decode(
                inputs, skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        elif job_type == 'token-to-id':
            return self.convert_tokens_to_ids(inputs)
        elif job_type == 'id-to-token':
            skip_special_tokens = kwargs.get('skip_special_tokens', False)
            return self.convert_ids_to_tokens(
                inputs, skip_special_tokens=skip_special_tokens)
        else:
            raise ValueError("Unrecognized job type.")

    def _tokenize(self, text: str) -> List[str]:
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                    text, never_split=self.all_special_tokens):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def save_vocabulary(self, save_directory: str) -> Tuple[str]:
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

    def _convert_token_to_id(self, token: str) -> int:
        r"""Converts a token (str/unicode) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        r"""Converts an index (integer) in a token (string/unicode) using
        the vocab.
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        r"""Converts a sequence of tokens (string) in a single string."""
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
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
            'never_split': None,
            '@no_typecheck': ['pretrained_model_name'],
        }
