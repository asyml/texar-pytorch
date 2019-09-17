# -*- coding: utf-8 -*-
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
Helper functions and classes for vocabulary processing.
"""
import warnings
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from texar.torch.utils.utils import (
    _recur_split, dict_lookup, str_join, strip_special_tokens)

__all__ = [
    "SpecialTokens",
    "Vocab",
    "map_ids_to_strs",
]


class SpecialTokens:
    r"""Special tokens, including :attr:`PAD`, :attr:`BOS`, :attr:`EOS`,
    :attr:`UNK`. These tokens will by default have token ids 0, 1, 2, 3,
    respectively.
    """
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"


def _make_defaultdict(keys: Sequence[Union[int, str]],
                      values: Sequence[Union[int, str]],
                      default_value: Union[int, str]) \
        -> DefaultDict[Union[int, str], Union[int, str]]:
    r"""Creates a Python `defaultdict`.

    Args:
        keys (list): Keys of the dictionary.
        values (list): Values correspond to keys. The two lists :attr:`keys` and
            :attr:`values` must be of the same length.
        default_value: default value returned when key is missing.

    Returns:
        defaultdict: A Python `defaultdict` instance that maps keys to values.
    """
    dict_: DefaultDict[Union[int, str], Union[int, str]]
    dict_ = defaultdict(lambda: default_value)
    for k, v in zip(keys, values):
        dict_[k] = v
    return dict_


class Vocab:
    r"""Vocabulary class that loads vocabulary from file, and maintains mapping
    tables between token strings and indexes.

    Each line of the vocab file should contains one vocabulary token, e.g.::

        vocab_token_1
        vocab token 2
        vocab       token | 3 .
        ...

    Args:
        filename (str): Path to the vocabulary file where each line contains
            one token.
        bos_token (str): A special token that will be added to the beginning of
            sequences.
        eos_token (str): A special token that will be added to the end of
            sequences.
        unk_token (str): A special token that will replace all unknown tokens
            (tokens not included in the vocabulary).
        pad_token (str): A special token that is used to do padding.
    """

    def __init__(self,
                 filename: str,
                 pad_token: str = SpecialTokens.PAD,
                 bos_token: str = SpecialTokens.BOS,
                 eos_token: str = SpecialTokens.EOS,
                 unk_token: str = SpecialTokens.UNK):
        self._filename = filename
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token

        self._id_to_token_map_py, self._token_to_id_map_py \
            = self.load(self._filename)

    def load(self, filename: str) \
            -> Tuple[Dict[int, str], Dict[str, int]]:
        r"""Loads the vocabulary from the file.

        Args:
            filename (str): Path to the vocabulary file.

        Returns:
            A tuple of mapping tables between word string and
            index, (:attr:`id_to_token_map_py`, :attr:`token_to_id_map_py`),
            where and :attr:`token_to_id_map_py` are python `defaultdict`
            instances.
        """
        with open(filename, "r") as vocab_file:
            vocab = list(line.strip() for line in vocab_file)

        warnings.simplefilter("ignore", UnicodeWarning)

        if self._bos_token in vocab:
            raise ValueError("Special begin-of-seq token already exists in the "
                             "vocabulary: '%s'" % self._bos_token)
        if self._eos_token in vocab:
            raise ValueError("Special end-of-seq token already exists in the "
                             "vocabulary: '%s'" % self._eos_token)
        if self._unk_token in vocab:
            raise ValueError("Special UNK token already exists in the "
                             "vocabulary: '%s'" % self._unk_token)
        if self._pad_token in vocab:
            raise ValueError("Special padding token already exists in the "
                             "vocabulary: '%s'" % self._pad_token)

        warnings.simplefilter("default", UnicodeWarning)

        # Places _pad_token at the beginning to make sure it take index 0.
        vocab = [self._pad_token, self._bos_token, self._eos_token,
                 self._unk_token] + vocab
        # Must make sure this is consistent with the above line
        vocab_size = len(vocab)

        # Creates python maps to interface with python code
        id_to_token_map_py = dict(zip(range(vocab_size), vocab))
        token_to_id_map_py = dict(zip(vocab, range(vocab_size)))

        return id_to_token_map_py, token_to_id_map_py

    def map_ids_to_tokens_py(self, ids: Union[List[int], np.ndarray]) \
            -> np.ndarray:
        r"""Maps ids into text tokens.

        The input :attr:`ids` and returned tokens are both python
        arrays or list.

        Args:
            ids: An `int` numpy array or (possibly nested) list of token ids.

        Returns:
            A numpy array of text tokens of the same shape as :attr:`ids`.
        """
        return dict_lookup(self.id_to_token_map_py, ids, self.unk_token)

    def map_tokens_to_ids_py(self, tokens: List[str]) -> np.ndarray:
        r"""Maps text tokens into ids.

        The input :attr:`tokens` and returned ids are both python
        arrays or list.

        Args:
            tokens: A numpy array or (possibly nested) list of text tokens.

        Returns:
            A numpy array of token ids of the same shape as :attr:`tokens`.
        """
        return dict_lookup(self.token_to_id_map_py, tokens, self.unk_token_id)

    @property
    def id_to_token_map_py(self) -> Dict[int, str]:
        r"""The dictionary instance that maps from token index to the string
        form.
        """
        return self._id_to_token_map_py

    @property
    def token_to_id_map_py(self) -> Dict[str, int]:
        r"""The dictionary instance that maps from token string to the index.
        """
        return self._token_to_id_map_py

    @property
    def size(self) -> int:
        r"""The vocabulary size.
        """
        return len(self.token_to_id_map_py)

    @property
    def bos_token(self) -> str:
        r"""A string of the special token indicating the beginning of sequence.
        """
        return self._bos_token

    @property
    def bos_token_id(self) -> int:
        r"""The `int` index of the special token indicating the beginning
        of sequence.
        """
        return self.token_to_id_map_py[self._bos_token]

    @property
    def eos_token(self) -> str:
        r"""A string of the special token indicating the end of sequence.
        """
        return self._eos_token

    @property
    def eos_token_id(self) -> int:
        r"""The `int` index of the special token indicating the end
        of sequence.
        """
        return self.token_to_id_map_py[self._eos_token]

    @property
    def unk_token(self) -> str:
        r"""A string of the special token indicating unknown token.
        """
        return self._unk_token

    @property
    def unk_token_id(self) -> int:
        r"""The `int` index of the special token indicating unknown token.
        """
        return self.token_to_id_map_py[self._unk_token]

    @property
    def pad_token(self) -> str:
        r"""A string of the special token indicating padding token. The
        default padding token is an empty string.
        """
        return self._pad_token

    @property
    def pad_token_id(self) -> int:
        r"""The `int` index of the special token indicating padding token.
        """
        return self.token_to_id_map_py[self._pad_token]

    @property
    def special_tokens(self) -> List[str]:
        r"""The list of special tokens
        [:attr:`pad_token`, :attr:`bos_token`, :attr:`eos_token`,
        :attr:`unk_token`].
        """
        return [self._pad_token, self._bos_token, self._eos_token,
                self._unk_token]


def map_ids_to_strs(ids: Union[np.ndarray, Sequence[int]],
                    vocab: Vocab,
                    join: bool = True, strip_pad: Optional[str] = '<PAD>',
                    strip_bos: Optional[str] = '<BOS>',
                    strip_eos: Optional[str] = '<EOS>') \
        -> Union[np.ndarray, List[str]]:
    r"""Transforms ``int`` indexes to strings by mapping ids to tokens,
    concatenating tokens into sentences, and stripping special tokens, etc.

    Args:
        ids: An n-D numpy array or (possibly nested) list of ``int`` indexes.
        vocab: An instance of :class:`~texar.torch.data.Vocab`.
        join (bool): Whether to concatenate along the last dimension of the
            the tokens into a string separated with a space character.
        strip_pad (str): The PAD token to strip from the strings (i.e., remove
            the leading and trailing PAD tokens of the strings). Default
            is ``"<PAD>"`` as defined in
            :class:`~texar.torch.data.SpecialTokens`.PAD.
            Set to `None` or `False` to disable the stripping.
        strip_bos (str): The BOS token to strip from the strings (i.e., remove
            the leading BOS tokens of the strings).
            Default is ``"<BOS>"`` as defined in
            :class:`~texar.torch.data.SpecialTokens`.BOS.
            Set to `None` or `False` to disable the stripping.
        strip_eos (str): The EOS token to strip from the strings (i.e., remove
            the EOS tokens and all subsequent tokens of the strings).
            Default is ``"<EOS>"`` as defined in
            :class:`~texar.torch.data.SpecialTokens`.EOS.
            Set to `None` or `False` to disable the stripping.

    Returns:
        If :attr:`join` is True, returns a `(n-1)`-D numpy array (or list) of
        concatenated strings. If :attr:`join` is False, returns an `n`-D numpy
        array (or list) of str tokens.

    Example:

        .. code-block:: python

            text_ids = [[1, 9, 6, 2, 0, 0], [1, 28, 7, 8, 2, 0]]

            text = map_ids_to_strs(text_ids, data.vocab)
            # text == ['a sentence', 'parsed from ids']

            text = map_ids_to_strs(
                text_ids, data.vocab, join=False,
                strip_pad=None, strip_bos=None, strip_eos=None)
            # text == [['<BOS>', 'a', 'sentence', '<EOS>', '<PAD>', '<PAD>'],
            #          ['<BOS>', 'parsed', 'from', 'ids', '<EOS>', '<PAD>']]
    """
    tokens = vocab.map_ids_to_tokens_py(ids)
    if isinstance(ids, (list, tuple)):
        tokens = tokens.tolist()

    str_ = str_join(tokens)

    str_ = strip_special_tokens(
        str_, strip_pad=strip_pad, strip_bos=strip_bos, strip_eos=strip_eos)

    if join:
        return str_
    else:
        return _recur_split(str_, ids)  # type: ignore
