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
Pre-trained XLNet Tokenizer.

Code structure adapted from:
    `https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_xlnet.py`
"""

from typing import Any, Dict, List, Optional, Tuple

import os
import unicodedata
from shutil import copyfile
import sentencepiece as spm

from texar.torch.modules.pretrained.pretrained_xlnet import PretrainedXLNetMixin
from texar.torch.data.tokenizers.pretrained_tokenizer_base import \
    PretrainedTokenizerBase

__all__ = [
    "XLNetTokenizer",
]

SPIECE_UNDERLINE = u'▁'


class XLNetTokenizer(PretrainedXLNetMixin, PretrainedTokenizerBase):
    r"""Pre-trained XLNet Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `xlnet-base-uncased`). Please refer to
            :class:`~texar.torch.modules.PretrainedXLNetMixin` for
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
        'xlnet-base-cased': None,
        'xlnet-large-cased': None,
    }
    _VOCAB_FILE_NAMES = {'vocab_file': 'spiece.model'}

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        self.load_pretrained_config(pretrained_model_name, cache_dir, hparams)

        super().__init__(hparams=None)

        self.__dict__: Dict

        self.config = {
            'do_lower_case': self.hparams['do_lower_case'],
            'remove_space': self.hparams['remove_space'],
            'keep_accents': self.hparams['keep_accents'],
        }

        if self.pretrained_model_dir is not None:
            vocab_file = os.path.join(self.pretrained_model_dir,
                                      self._VOCAB_FILE_NAMES['vocab_file'])
            assert pretrained_model_name is not None
            if self._MAX_INPUT_SIZE.get(pretrained_model_name):
                self.max_len = self._MAX_INPUT_SIZE[pretrained_model_name]
        else:
            vocab_file = self.hparams['vocab_file']
            if self.hparams.get('max_len'):
                self.max_len = self.hparams['max_len']

        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path "
                             "'{}".format(vocab_file))

        self.do_lower_case = self.hparams["do_lower_case"]
        self.remove_space = self.hparams["remove_space"]
        self.keep_accents = self.hparams["keep_accents"]
        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

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

    def _preprocess_text(self, inputs: str) -> str:
        r"""Pre-process the text, including removing space,
        stripping accents, and lower-casing the text.
        """
        if self.remove_space:
            outputs = ' '.join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = ''.join([c for c in outputs if not
            unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _map_text_to_token(self, text: str,  # type: ignore
                           sample: bool = False) -> List[str]:
        text = self._preprocess_text(text)
        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)

        new_pieces: List[str] = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and \
                        cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def save_vocab(self, save_directory: str) -> Tuple[str]:
        r"""Save the sentencepiece vocabulary (copy original file) to
        a directory.
        """
        if not os.path.isdir(save_directory):
            raise ValueError("Vocabulary path ({}) should be a "
                             "directory".format(save_directory))
        out_vocab_file = os.path.join(save_directory,
                                      self._VOCAB_FILE_NAMES['vocab_file'])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    @property
    def vocab_size(self) -> int:
        return len(self.sp_model)

    def _map_token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def _map_id_to_token(self, index: int) -> str:
        token = self.sp_model.IdToPiece(index)
        return token

    def map_token_to_text(self, tokens: List[str]) -> str:
        r"""Maps a sequence of tokens (string) in a single string."""
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
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
                "pretrained_model_name": "xlnet-base-cased",
                "vocab_file": None,
                "max_len": None,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "sep_token": "<sep>",
                "pad_token": "<pad>",
                "cls_token": "<cls>",
                "mask_token": "<mask>",
                "additional_special_tokens": ["<eop>", "<eod>"],
                "do_lower_case": False,
                "remove_space": True,
                "keep_accents": False,
            }

        Here:

        `"pretrained_model_name"`: str or None
            The name of the pre-trained XLNet model.

        `"vocab_file"`: str or None
            The path to a sentencepiece vocabulary file.

        `"max_len"`: int or None
            The maximum sequence length that this model might ever be used with.

        `"bos_token"`: str
            Beginning of sentence token.

        `"eos_token"`: str
            End of sentence token.

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

        `"additional_special_tokens"`: list
            A list of additional special tokens.

        `"do_lower_case"`: bool
            Whether to lower-case the text.

        `"remove_space"`: bool
            Whether to remove the space in the text.

        `"keep_accents"`: bool
            Whether to keep the accents in the text.
        """
        return {
            'pretrained_model_name': 'xlnet-base-cased',
            'vocab_file': None,
            'max_len': None,
            'bos_token': '<s>',
            'eos_token': '</s>',
            'unk_token': '<unk>',
            'sep_token': '<sep>',
            'pad_token': '<pad>',
            'cls_token': '<cls>',
            'mask_token': '<mask>',
            'additional_special_tokens': ['<eop>', '<eod>'],
            'do_lower_case': False,
            'remove_space': True,
            'keep_accents': False,
            '@no_typecheck': ['pretrained_model_name'],
        }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str):
        r"""Returns the configuration of the pre-trained XLNet tokenizer."""
        return {
            'vocab_file': None,
            'max_len': None,
            'bos_token': '<s>',
            'eos_token': '</s>',
            'unk_token': '<unk>',
            'sep_token': '<sep>',
            'pad_token': '<pad>',
            'cls_token': '<cls>',
            'mask_token': '<mask>',
            'additional_special_tokens': ['<eop>', '<eod>'],
            'do_lower_case': False,
            'remove_space': True,
            'keep_accents': False,
        }
