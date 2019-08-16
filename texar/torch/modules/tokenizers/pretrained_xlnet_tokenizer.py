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

The code structure adapted from:
    `https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_xlnet.py`
"""

from typing import Any, Dict, List, Optional, Tuple

import os
import unicodedata
import sentencepiece as spm
from shutil import copyfile

from texar.torch.modules.tokenizers.pretrained_tokenizer_base import \
    PretrainedTokenizerBase

__all__ = [
    "PretrainedXLNetTokenizer",
]

_XLNET_PATH = "https://storage.googleapis.com/xlnet/released_models/"

SPIECE_UNDERLINE = u'▁'


class PretrainedXLNetTokenizer(PretrainedTokenizerBase):
    r"""Pre-trained XLNet Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `xlnet-base-uncased`). Please refer to
            :class:`~texar.torch.modules.pretrained.PretrainedXLNetMixin` for
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

    _MODEL_NAME = "XLNet"
    _MODEL2URL = {
        'xlnet-base-cased':
            _XLNET_PATH + "cased_L-12_H-768_A-12.zip",
        'xlnet-large-cased':
            _XLNET_PATH + "cased_L-24_H-1024_A-16.zip",
    }
    _MAX_INPUT_SIZE = {
        'xlnet-base-cased': None,
        'xlnet-large-cased': None,
    }
    _VOCAB_FILE_NAMES = {'vocab_file': 'spiece.model'}

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        super().__init__(hparams=hparams)

        self.load_pretrained_tokenizer(pretrained_model_name, cache_dir)

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

        self.do_lower_case = self.hparams["do_lower_case"]
        self.remove_space = self.hparams["remove_space"]
        self.keep_accents = self.hparams["keep_accents"]
        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def forward(self, inputs, job_type, **kwargs):
        if job_type == 'text-to-token':
            sample = kwargs.get('sample', False)
            return self.tokenize(inputs, sample=sample)
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs: str) -> str:
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

    def _tokenize(self, text: str, sample: bool = False) -> List[str]:
        text = self.preprocess_text(text)
        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)

        new_pieces = []
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

    def save_vocabulary(self, save_directory: str) -> Tuple[str]:
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

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
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
