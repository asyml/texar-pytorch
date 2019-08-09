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

The code structure adapted from:
    `https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_bert.py`
"""

from typing import Any, Dict, List, Optional, Tuple

import os

from texar.torch.modules.tokenizers.pretrained_tokenizer_base import \
    PretrainedTokenizerBase
from texar.torch.modules.tokenizers.pretrained_bert_tokenizer_utils import \
    load_vocab, BasicTokenizer, WordpieceTokenizer

__all__ = [
    'PretrainedBERTTokenizer',
]


_BERT_PATH = "https://storage.googleapis.com/bert_models/"


class PretrainedBERTTokenizer(PretrainedTokenizerBase):
    r"""Pre-trained BERT Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `bert-base-uncased`). Please refer to
            :class:`~texar.torch.modules.pretrained.PretrainedBERTMixin` for
            all supported models (including the standard BERT models and
            variants like RoBERTa).
            If None, the model name in :attr:hparams is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    _MODEL_NAME = "BERT"
    _MODEL2URL = {
        'bert-base-uncased':
            _BERT_PATH + "2018_10_18/uncased_L-12_H-768_A-12.zip",
        'bert-large-uncased':
            _BERT_PATH + "2018_10_18/uncased_L-24_H-1024_A-16.zip",
        'bert-base-cased':
            _BERT_PATH + "2018_10_18/cased_L-12_H-768_A-12.zip",
        'bert-large-cased':
            _BERT_PATH + "2018_10_18/cased_L-24_H-1024_A-16.zip",
        'bert-base-multilingual-uncased':
            _BERT_PATH + "2018_11_23/multi_cased_L-12_H-768_A-12.zip",
        'bert-base-multilingual-cased':
            _BERT_PATH + "2018_11_03/multilingual_L-12_H-768_A-12.zip",
        'bert-base-chinese':
            _BERT_PATH + "2018_11_03/chinese_L-12_H-768_A-12.zip",
    }
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

        self.load_pretrained_tokenizer(pretrained_model_name, cache_dir)

        if self.pretrained_model_dir is not None:
            vocab_file = os.path.join(self.pretrained_model_dir, 'vocab.txt')
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
            save_directory = os.path.join(save_directory, 'vocab.txt')
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
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, id: int) -> str:
        return self.ids_to_tokens.get(id, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            'pretrained_model_name': 'bert-base-uncased',
            'max_len': 512,
            'vocab_file': None,
            'do_lower_case': True,
            'do_basic_tokenize': True,
            'never_split': None,
            'unk_token': '[UNK]',
            'sep_token': '[SEP]',
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'mask_token': '[MASK]',
            'tokenize_chinese_chars': True,
            '@no_typecheck': ['pretrained_model_name'],
        }
