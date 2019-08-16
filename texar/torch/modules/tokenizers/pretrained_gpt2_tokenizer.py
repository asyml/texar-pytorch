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
Pre-trained GPT-2 Tokenizer.

The code structure adapted from:
    `https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_gpt2.py`
"""

from typing import Any, Dict, List, Optional, Tuple

import os
import json
import regex as re

from texar.torch.modules.tokenizers.pretrained_tokenizer_base import \
    PretrainedTokenizerBase
from texar.torch.modules.tokenizers.pretrained_gpt2_tokenizer_utils import \
    bytes_to_unicode, get_pairs

__all__ = [
    'PretrainedGPT2Tokenizer',
]

_GPT2_PATH = "https://storage.googleapis.com/gpt-2/models/"
_CHECKPOINT_FILES = [
    "checkpoint", "encoder.json", "hparams.json", "vocab.bpe",
    "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta"]


class PretrainedGPT2Tokenizer(PretrainedTokenizerBase):
    r"""Pre-trained BERT Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `117M`). Please refer to
            :class:`~texar.torch.modules.pretrained.PretrainedGPT2Mixin` for
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
    _MODEL_NAME = "GPT2"
    _MODEL2URL = {
        '117M': [_GPT2_PATH + f"117M/{file}" for file in _CHECKPOINT_FILES],
        '345M': [_GPT2_PATH + f"345M/{file}" for file in _CHECKPOINT_FILES],
    }
    _MAX_INPUT_SIZE = {
        '117M': 1024,
        '345M': 1024,
    }
    _VOCAB_FILE_NAMES = {
        'vocab_file': 'vocab.json',
        'merges_file': 'merges.txt',
    }

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        super().__init__(hparams=hparams)

        self.load_pretrained_tokenizer(pretrained_model_name, cache_dir)

        if self.pretrained_model_dir is not None:
            vocab_file = os.path.join(self.pretrained_model_dir,
                                      self._VOCAB_FILE_NAMES['vocab_file'])
            merges_file = os.path.join(self.pretrained_model_dir,
                                       self._VOCAB_FILE_NAMES['merges_file'])
            if self._MAX_INPUT_SIZE.get(pretrained_model_name):
                self.max_len = self._MAX_INPUT_SIZE[pretrained_model_name]
        else:
            vocab_file = self.hparams['vocab_file']
            merges_file = self.hparams['merges_file']
            if self.hparams.get('max_len'):
                self.max_len = self.hparams['max_len']

        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path "
                             "'{}".format(vocab_file))

        if not os.path.isfile(merges_file):
            raise ValueError("Can't find a merges file at path "
                             "'{}".format(merges_file))

        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = self.hparams["errors"]  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| 
            ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def save_vocabulary(self, save_directory: str) -> Tuple[str, str]:
        r"""Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            raise ValueError("Vocabulary path ({}) should be a "
                             "directory".format(save_directory))

        vocab_file = os.path.join(save_directory,
                                  self._VOCAB_FILE_NAMES['vocab_file'])
        merge_file = os.path.join(save_directory,
                                  self._VOCAB_FILE_NAMES['merges_file'])

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(),
                                                  key=lambda kv: kv[1]):
                if index != token_index:
                    print("Saving vocabulary to {}: BPE merge indices are "
                          "not consecutive. Please check that the tokenizer "
                          "is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        return (vocab_file, merge_file)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def _convert_token_to_id(self, token: str) -> int:
        r"""Converts a token (str/unicode) in an id using the vocab."""
        if token in self.encoder:
            return self.encoder.get(token)
        return self.encoder.get(self.unk_token)

    def _convert_id_to_token(self, index: int) -> str:
        r"""Converts an index (integer) in a token (string/unicode) using
        the vocab.
        """
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        r"""Converts a sequence of tokens (string) in a single string."""
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors)
        return text

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            'pretrained_model_name': '117M',
            'vocab_file': None,
            'merges_file': None,
            'max_len': 1024,
            'bos_token': '<|endoftext|>',
            'eos_token': '<|endoftext|>',
            'unk_token': '<unk>',
            'errors': 'replace',
            '@no_typecheck': ['pretrained_model_name'],
        }
