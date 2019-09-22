"""
Various utilities for data module
"""

from enum import Enum
from typing import (
    Any, Dict, ItemsView, KeysView, List, Optional, Tuple, Union, ValuesView)

import numpy as np

__all__ = [
    'padded_batch',
    'connect_name',
    'Batch',
    '_LazyStrategy',
    '_CacheStrategy',
]


def padded_batch(examples: Union[List[np.ndarray], List[List[int]]],
                 pad_length: Optional[int] = None, pad_value: int = 0) \
        -> Tuple[np.ndarray, List[int]]:
    r"""Pad a batch of integer lists (or numpy arrays) to the same length, and
    stack them together.

    Args:
        examples (list of lists): The list of examples.
        pad_length (int, optional): The desired length after padding. If
            `None`, use the maximum length of lists in the batch. Defaults to
            `None`. Note that if ``pad_length`` is not `None` and the
            maximum length of lists is greater than ``pad_length``, then all
            lists are padded to the maximum length instead.
        pad_value (int, optional): The value to fill in the padded positions.
            Defaults to 0.

    Returns:
        A tuple of two elements, with the first being the padded batch, and the
        second being the original lengths of each list.
    """
    lengths = [len(sent) for sent in examples]
    pad_length = pad_length or max(lengths)

    padded = np.full((len(examples), pad_length), pad_value, dtype=np.int64)
    for b_idx, sent in enumerate(examples):
        length = lengths[b_idx]
        padded[b_idx, :length] = sent[:length]
    return padded, lengths


def connect_name(lhs_name, rhs_name):
    if not lhs_name:
        return rhs_name
    if not rhs_name:
        return lhs_name
    return "{}_{}".format(lhs_name, rhs_name)


class Batch:
    r"""Wrapper over Python dictionaries representing a batch. It provides a
    dictionary-like interface to access its fields. This class can be used in
    the followed way

    .. code-block:: python

        hparams = {
            'dataset': { 'files': 'data.txt', 'vocab_file': 'vocab.txt' },
            'batch_size': 1
        }

        data = MonoTextData(hparams)
        iterator = DataIterator(data)

        for batch in iterator:
            # batch is Batch object and contains the following fields
            # batch == {
            #    'text': [['<BOS>', 'example', 'sequence', '<EOS>']],
            #    'text_ids': [[1, 5, 10, 2]],
            #    'length': [4]
            # }

            input_ids = torch.tensor(batch['text_ids'])

            # we can also access the elements using dot notation
            input_text = batch.text
    """

    def __init__(self, batch_size: int, batch: Optional[Dict[str, Any]] = None,
                 **kwargs):
        self.batch_size = batch_size
        self._batch = batch or {}
        if isinstance(self._batch, dict):
            self._batch.update(kwargs)

    def __getattr__(self, item):
        if item not in super().__getattribute__('_batch'):
            raise AttributeError
        return self._batch[item]

    def __getitem__(self, item):
        return self._batch[item]

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self):
        return repr(self._batch)

    def keys(self) -> KeysView[str]:
        return self._batch.keys()

    def values(self) -> ValuesView[Any]:
        return self._batch.values()

    def items(self) -> ItemsView[str, Any]:
        return self._batch.items()


class _LazyStrategy(Enum):
    NONE = "none"
    PROCESS = "process"
    ALL = "all"


class _CacheStrategy(Enum):
    NONE = "none"
    LOADED = "loaded"
    PROCESSED = "processed"
