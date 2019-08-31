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
SentencePiece Tokenizer.
"""

from typing import Any, Dict, List, Optional, Tuple

import os
from shutil import copyfile, move

import sentencepiece as spm

from texar.torch.data.tokenizers.pretrained_tokenizer_base import \
    PretrainedTokenizerBase
from texar.torch.modules.pretrained.pretrained_base import default_download_dir
from texar.torch.utils.utils_io import maybe_create_dir

__all__ = [
    "SentencePieceTokenizer"
]


class SentencePieceTokenizer(PretrainedTokenizerBase):
    r"""SentencePiece Tokenizer.

    Args:
        cache_dir (optional): the path to a folder in which the
            trained `sentencepiece` model will be cached. If `None` (default),
            a default directory (user's home directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """
    _IS_PRETRAINED = False
    _VOCAB_FILE_NAMES = {
        'vocab_file': 'spiece.model',
    }
    _TRAIN_ARG_MAP = {
        'text_file': 'input',
        'model_type': 'model_type',
        'vocab_size': 'vocab_size',
        'bos_token': 'bos_piece',
        'eos_token': 'eos_piece',
        'unk_token': 'unk_piece',
        'pad_token': 'pad_piece',
    }

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        super().__init__(hparams=hparams)

        self.__dict__: Dict

        if self.hparams['vocab_file'] is not None:
            self.vocab_file = self.hparams['vocab_file']
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(self.vocab_file)

            bos_id = self.sp_model.bos_id()
            eos_id = self.sp_model.eos_id()
            unk_id = self.sp_model.unk_id()
            pad_id = self.sp_model.pad_id()

            self.bos_token = None
            if bos_id != -1:
                self.bos_token = self.sp_model.IdToPiece(bos_id)

            self.eos_token = None
            if eos_id != -1:
                self.eos_token = self.sp_model.IdToPiece(eos_id)

            self.unk_token = None
            if unk_id != -1:
                self.unk_token = self.sp_model.IdToPiece(unk_id)

            self.pad_token = None
            if pad_id != -1:
                self.pad_token = self.sp_model.IdToPiece(pad_id)

        elif self.hparams['text_file'] is not None:
            cmd = ['--model_prefix=spiece']
            for arg, val in self.hparams.items():
                if arg in self._TRAIN_ARG_MAP:
                    cmd.append('--' + self._TRAIN_ARG_MAP[arg] + '=' + str(val))

            cache_path = self.train(" ".join(cmd), cache_dir)
            self.vocab_file = os.path.join(
                cache_path, self._VOCAB_FILE_NAMES['vocab_file'])
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(self.vocab_file)
        else:
            raise ValueError("'vocab_file' and 'text_file' can not be None "
                             "at the same time.")

    @classmethod
    def train(cls, cmd: str,  # type: ignore
              cache_dir: Optional[str] = None) -> str:
        if cache_dir is None:
            cache_path = str(default_download_dir('SentencePiece'))
        else:
            if not os.path.isdir(cache_dir):
                raise ValueError(f"Cache directory ({cache_dir}) should be a "
                                 f"directory.")
            cache_path = os.path.abspath(cache_dir)

        maybe_create_dir(cache_path)

        spm.SentencePieceTrainer.Train(cmd)
        cwd = os.getcwd()

        vocab_file = os.path.join(cwd, cls._VOCAB_FILE_NAMES['vocab_file'])
        out_vocab_file = os.path.join(
            cache_path, cls._VOCAB_FILE_NAMES['vocab_file'])

        if os.path.abspath(vocab_file) != os.path.abspath(out_vocab_file):
            move(vocab_file, out_vocab_file)

        # Delete spiece.vocab (We might want to keep it as well)
        extra_file = vocab_file.rstrip('model') + 'vocab'
        os.remove(extra_file)

        return cache_path

    # spm.SentencePieceProcessor() is a SwigPyObject object which cannot be
    # pickled. We need to define __getstate__ here.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["vocab_file"] = None
        return state, self.vocab_file

    # spm.SentencePieceProcessor() is a SwigPyObject object which cannot be
    # pickled. We need to define __setstate__ here.
    def __setstate__(self, d):
        self.__dict__, self.vocab_file = d
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def save_vocab(self, save_directory: str) -> Tuple[str]:
        r"""Save the sentencepiece vocabulary (copy original file) to
        a directory.
        """
        if not os.path.isdir(save_directory):
            raise ValueError("Vocabulary path ({}) should be a "
                             "directory".format(save_directory))
        out_vocab_file = os.path.join(
            save_directory, self._VOCAB_FILE_NAMES['vocab_file'])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    @property
    def vocab_size(self) -> int:
        return len(self.sp_model)

    def _map_text_to_token(self, text: str) -> List[str]:  # type: ignore
        return self.sp_model.EncodeAsPieces(text)

    def _map_token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def _map_id_to_token(self, index: int) -> str:
        token = self.sp_model.IdToPiece(index)
        return token

    def map_token_to_text(self, tokens: List[str]) -> str:
        r"""Maps a sequence of tokens (string) in a single string."""
        out_string = self.sp_model.DecodePieces(tokens)
        return out_string

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        * The tokenizer is determined by `hparams['vocab_file']` if it's
          specified. In this case, all other configurations in `hparams`
          are ignored.
        * Otherwise, the tokenizer is trained based on `hparams['text_file']`.
        * `hparams['vocab_file']` and `hparams['text_file']` can not be None
        at the same time.

        .. code-block:: python

            {
                "vocab_file": None,
                "text_file": None,
                "vocab_size": None,
                "model_type": "unigram",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }

        Here:

        `"vocab_file"`: str or None
            The path to a sentencepiece vocabulary file.

        `"text_file"`: str or None
            Comma separated list of input sentences.

        `"vocab_size"`: int or None
            Vocabulary size.

        `"model_type"`: str
            Model algorithm to train the tokenizer. Available algorithms are:
            ``unigram``, ``bpe``, ``word``, and ``char``.

        `"bos_token"`: str
            Beginning of sentence token.

        `"eos_token"`: str
            End of sentence token.

        `"unk_token"`: str
            Unknown token.

        `"pad_token"`: str
            Padding token.
        """
        return {
            'vocab_file': None,
            'text_file': None,
            'vocab_size': None,
            'model_type': 'unigram',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'pad_token': '<pad>',
            'unk_token': '<unk>',
        }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
        pass

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str, **kwargs):
        pass
