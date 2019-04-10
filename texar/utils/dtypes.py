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

# pylint: disable=invalid-name, no-member, protected-access

from typing import Any, Dict, Optional, Union

import numpy as np

from texar.hyperparams import HParams

__all__ = [
    'is_str',
    'is_callable',
    'maybe_hparams_to_dict',
]


def is_callable(x):
    r"""Return `True` if :attr:`x` is callable.
    """
    return callable(x)


def is_str(x):
    r"""Returns `True` if :attr:`x` is either a str or unicode. Returns `False`
    otherwise.
    """
    return isinstance(x, str)


def maybe_hparams_to_dict(hparams: Optional[Union[HParams, Dict[str, Any]]]) \
        -> Optional[Dict[str, Any]]:
    r"""If :attr:`hparams` is an instance of :class:`~texar.HParams`,
    converts it to a `dict` and returns. If :attr:`hparams` is a `dict`,
    returns as is.
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

    Adapted from `tensorflow.compat.as_text`.

    Args:
        bytes_or_text: A `bytes`, `str`, or `unicode` object.
            encoding: A string indicating the charset for decoding unicode.

    Returns:
        A `unicode` (Python 2) or `str` (Python 3) object.

    Raises:
        TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, str):
        return bytes_or_text
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text.decode(encoding)
    else:
        raise TypeError(
            f"Expected binary or unicode string, got {bytes_or_text!r}")
