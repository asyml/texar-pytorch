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

from texar.torch.data.tokenizers.tokenizer_base import TokenizerBase
from texar.torch.modules.pretrained.pretrained_base import default_download_dir
from texar.torch.utils.utils_io import maybe_create_dir

__all__ = [
    "SentencePieceTokenizer"
]


class SentencePieceTokenizer(TokenizerBase):
    r"""SentencePiece Tokenizer. This class is a wrapper of Google's
    `SentencePiece`_ with richer ready-to-use functionalities such as
    adding tokens and saving/loading.

    `SentencePiece` is an unsupervised text tokenizer mainly for Neural
    Network-based text generation systems where the vocabulary size
    is predetermined prior to the neural model training. `SentencePiece`
    implements sub-word units (e.g., byte-pair-encoding (BPE) and unigram
    language model) with the extension of direct training from raw sentences.

    The supported algorithms in `SentencePiece` are: ``bpe``, ``word``,
    ``char``, and ``unigram``, which is specified in :attr:`hparams`.

    Args:
        cache_dir (optional): the path to a folder in which the
            trained `sentencepiece` model will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.

    .. _`SentencePiece`: https://github.com/google/sentencepiece
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
                    if arg in self._SPECIAL_TOKENS_ATTRIBUTES and val is None:
                        cmd.append('--' + arg.replace('token', 'id') + '=-1')
                    else:
                        cmd.append('--' + self._TRAIN_ARG_MAP[arg] + '=' +
                                   str(val))

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
        r"""Trains the tokenizer from the raw text file. This function is
        a wrapper of `sentencepiece.SentencePieceTrainer.Train`_ function.

        Example:

        .. code-block:: python

            SentencePieceTokenizer.train('--input=test/botchan.txt
            --model_prefix=m --vocab_size=1000')

        Args:
            cmd (str): the command for the tokenizer training procedure.
                See ``sentencepiece.SentencePieceTrainer.Train`` for the
                detailed usage.
            cache_dir (optional): the path to a folder in which the trained
                `sentencepiece` model will be cached. If `None` (default),
                a default directory (`texar_pytorch` folder under user's home
                directory) will be used.

        Returns:
            Path to the cache directory.

        .. _`sentencepiece.SentencePieceTrainer.Train`:
            https://github.com/google/sentencepiece/blob/master/python/sentencepiece.py
        """
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

    def save_vocab(self, save_dir: str) -> Tuple[str]:
        r"""Save the sentencepiece vocabulary (copy original file) to
        a directory.
        """
        if not os.path.isdir(save_dir):
            raise ValueError("Vocabulary path ({}) should be a "
                             "directory".format(save_dir))
        out_vocab_file = os.path.join(
            save_dir, self._VOCAB_FILE_NAMES['vocab_file'])

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

        * If `hparams['vocab_file']` is specified, the tokenizer is directly
          loaded from the vocabulary file. In this case, all other
          configurations in `hparams` are ignored.
        * Otherwise, the tokenizer is automatically trained based on
          `hparams['text_file']`. In this case, `hparams['vocab_size']` must
          be specified.
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
            Vocabulary size. The user can specify the vocabulary size, and the
            tokenizer training procedure will train and yield a vocabulary
            of the specified size.

        `"model_type"`: str
            Model algorithm to train the tokenizer. Available algorithms are:
            ``bpe``, ``word``, ``char``, and ``unigram``.

        `"bos_token"`: str or None
            Beginning of sentence token. Set None to disable ``bos_token``.

        `"eos_token"`: str or None
            End of sentence token. Set None to disable ``eos_token``.

        `"unk_token"`: str or None
            Unknown token. Set None to disable ``unk_token``.

        `"pad_token"`: str or None
            Padding token. Set None to disable ``pad_token``.
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
