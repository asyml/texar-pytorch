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
Utility functions related to data types.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from texar.torch.hyperparams import HParams

__all__ = [
    'torch_bool',
    'get_numpy_dtype',
    'is_str',
    'is_callable',
    'get_supported_scalar_types',
    'maybe_hparams_to_dict',
    'compat_as_text',
]

# `torch.bool` exists in PyTorch 1.1, but the default type for comparisons
# is still `torch.uint8`.
torch_bool = (torch.empty(()) < 0).dtype

DTYPE_MAP = {
    np.float32: ['float32', 'float', 'tf.float32', 'torch.float',
                 'torch.float32', float, np.float32, torch.float32],
    np.float64: ['float64', 'tf.float64', 'torch.float64', np.float64,
                 np.float_, torch.float64],
    np.float16: ['float16', 'tf.float16', 'torch.float16', np.float16,
                 torch.float16],
    np.int32: ['int', 'int32', 'tf.int32', 'torch.int', 'torch.int32', int,
               np.int32, torch.int32],
    np.int64: ['int64', 'tf.int64', 'torch.int64', np.int64, np.int_,
               torch.int64],
    np.int16: ['int16', 'tf.int16', 'torch.int16', np.int16, torch.int16],
    np.int8: ['int8', 'char', 'tf.int8', 'torch.int8', np.int8, torch.int8],
    np.uint8: ['uint8', 'tf.uint8', 'torch.uint8', np.uint8, torch.uint8],
    np.bool_: ['bool', 'tf.bool', 'torch.bool', bool, np.bool, np.bool_,
               torch_bool],
    np.str_: ['string', 'str', 'tf.string', str, np.str, np.str_],
    np.bytes_: ['bytes', 'np.bytes', bytes, np.bytes_]
}


def get_numpy_dtype(dtype: Union[str, type]):
    r"""Returns equivalent NumPy dtype.

    Args:
        dtype: A str, Python numeric or string type, NumPy data type, or
            PyTorch dtype.

    Returns:
        The corresponding NumPy dtype.
    """
    for np_dtype, valid_values in DTYPE_MAP.items():
        if dtype in valid_values:
            return np_dtype

    raise ValueError(
        f"Unsupported conversion from type {dtype!s} to NumPy dtype")


def is_callable(x):
    r"""Return `True` if :attr:`x` is callable.
    """
    return callable(x)


def is_str(x):
    r"""Returns `True` if :attr:`x` is either a str or unicode.
    Returns `False` otherwise.
    """
    return isinstance(x, str)


def get_supported_scalar_types():
    r"""Returns a list of scalar types supported.
    """
    types = []
    for key, value in DTYPE_MAP.items():
        if key not in {np.str_, np.bytes_}:
            types.extend(value)

    return types


def maybe_hparams_to_dict(hparams: Optional[Union[HParams, Dict[str, Any]]]) \
        -> Optional[Dict[str, Any]]:
    r"""If :attr:`hparams` is an instance of :class:`~texar.torch.HParams`,
    converts it to a ``dict`` and returns. If :attr:`hparams` is a ``dict``,
    returns as is.

    Args:
        hparams: The :class:`~texar.torch.HParams` instance to convert.

    Returns:
        dict: The corresponding ``dict`` instance
    """
    if hparams is None:
        return None
    if isinstance(hparams, dict):
        return hparams
    return hparams.todict()


def _maybe_list_to_array(str_list, dtype_as):
    if isinstance(dtype_as, (list, tuple)):
        return type(dtype_as)(str_list)
    elif isinstance(dtype_as, np.ndarray):
        return np.array(str_list)
    else:
        return str_list


def _as_text(bytes_or_text, encoding='utf-8'):
    r"""Returns the given argument as a unicode string.

    Adapted from ``tensorflow.compat.as_text``.

    Args:
        bytes_or_text: A ``bytes``, ``str``, or ``unicode`` object.
            encoding: A string indicating the charset for decoding unicode.

    Returns:
        A ``unicode`` (Python 2) or ``str`` (Python 3) object.

    Raises:
        TypeError: If ``bytes_or_text`` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, str):
        return bytes_or_text
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text.decode(encoding)
    else:
        raise TypeError(
            f"Expected binary or unicode string, got {bytes_or_text!r}")


def compat_as_text(str_):
    r"""Converts strings into ``unicode`` (Python 2) or ``str`` (Python 3).

    Args:
        str\_: A string or other data types convertible to string, or an
            `n`-D numpy array or (possibly nested) list of such elements.

    Returns:
        The converted strings of the same structure/shape as :attr:`str_`.
    """

    def _recur_convert(s):
        if isinstance(s, (list, tuple, np.ndarray)):
            s_ = [_recur_convert(si) for si in s]
            return _maybe_list_to_array(s_, s)
        else:
            try:
                return _as_text(s)
            except TypeError:
                return _as_text(str(s))

    text = _recur_convert(str_)

    return text
