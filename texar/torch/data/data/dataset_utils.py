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
    'FieldBatch',
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
    r"""Wrapper over Python dictionaries representing a batch. This provides a
    common interface with :class:`~texar.torch.data.data.dataset_utils.Batch`
    that allows accessing via attributes.
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

    def keys(self) -> KeysView[str]:
        return self._batch.keys()

    def values(self) -> ValuesView[Any]:
        return self._batch.values()

    def items(self) -> ItemsView[str, Any]:
        return self._batch.items()


class FieldBatch(Batch):
    r"""Defines a batch of examples with support for multiple fields. This is
    a simplified version of `torchtext.data.Batch`, with all the useless stuff
    removed.
    """

    def __init__(self, data=None, dataset=None, device=None):
        r"""Create a Batch from a list of examples.
        """
        if data is not None:
            batch_size = len(data)
            _batch_dict = {}
            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    _batch_dict[name] = field.process(batch, device=device)
            super().__init__(batch_size, _batch_dict)
        else:
            super().__init__(0)


class _LazyStrategy(Enum):
    NONE = "none"
    PROCESS = "process"
    ALL = "all"


class _CacheStrategy(Enum):
    NONE = "none"
    LOADED = "loaded"
    PROCESSED = "processed"
