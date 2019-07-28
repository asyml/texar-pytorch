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
Utility functions related to tensor shapes.
"""

from typing import Any, List, Optional, Union

import numpy as np
import torch

from texar.torch.utils import utils
from texar.torch.utils.types import MaybeList

__all__ = [
    "transpose_batch_time",
    "get_batch_size",
    "get_rank",
    "mask_sequences",
    "flatten",
    "pad_and_concat",
]


def transpose_batch_time(inputs: torch.Tensor) -> torch.Tensor:
    r"""Transposes inputs between time-major and batch-major.

    Args:
        inputs: A Tensor of shape ``[batch_size, max_time, ...]`` (batch-major)
            or ``[max_time, batch_size, ...]`` (time-major), or a (possibly
            nested) tuple of such elements.

    Returns:
        A (possibly nested tuple of) Tensor with transposed batch and
        time dimensions of inputs.
    """
    return inputs.transpose(0, 1)


def get_batch_size(tensor: torch.Tensor) -> int:
    r"""Returns an ``int`` representing the batch size, i.e.,
    the size of the 1st dimension of :attr:`tensor`.
    """
    return tensor.size(0)


def get_rank(tensor: torch.Tensor) -> int:
    r"""Returns the tensor rank as a Python ``int``. The input tensor can also
    be a Python array.

    Args:
        tensor: A Tensor or Python array.

    Returns:
        A Python ``int`` representing the rank of :attr:`tensor`. Returns
        `None` if the rank cannot be determined.
    """
    if torch.is_tensor(tensor):
        rank = tensor.dim()
    else:
        array = np.asarray(tensor)
        rank = array.ndim
    return rank


def mask_sequences(sequence: Union[torch.Tensor, List[int]],
                   sequence_length: Union[torch.LongTensor, List[int]],
                   dtype: Optional[torch.dtype] = None,
                   time_major: bool = False) -> torch.Tensor:
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    :attr:`sequence` and :attr:`sequence_length` can either be python
    arrays or Tensors, respectively. If both are Python arrays (or None), the
    return will be a Python array as well.

    Args:
        sequence: A Tensor or Python array of sequence values.
            If ``time_major==False`` (default), this must be a Tensor of shape
            ``[batch_size, max_time, ...]``. The batch and time dimension is
            exchanged if ``time_major==True``.
        sequence_length: A Tensor or python array of shape ``[batch_size]``.
            Time steps beyond the respective sequence lengths will be
            made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            ``[max_time, batch_size, ...]``.
            If `False` (default), :attr:`sequence` must have
            shape ``[batch_size, max_time, ...]``.

    Returns:
        The masked sequence, i.e., a Tensor or python array of the same shape
        as :attr:`sequence` but with masked-out entries (set to zero).

        If both :attr:`sequence` and :attr:`sequence_length` are python
        arrays, the returned value is a python array as well.
    """
    if not torch.is_tensor(sequence):
        sequence = torch.tensor(sequence, dtype=dtype)
    sequence: torch.Tensor

    rank = sequence.dim()
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")

    if time_major:
        sequence = transpose_batch_time(sequence)
    max_time = sequence.size(1)
    if dtype is None:
        dtype = sequence.dtype
    mask = utils.sequence_mask(sequence_length, max_time, dtype=dtype)
    mask = mask.view(*mask.size(), *([1] * (rank - 2)))
    sequence = sequence * mask
    if time_major:
        sequence = transpose_batch_time(sequence)

    return sequence


def flatten(tensor: torch.Tensor, preserve_dims: int,
            flattened_dim: Optional[int] = None) -> torch.Tensor:
    r"""Flattens a tensor whiling keeping several leading dimensions.

    :attr:`preserve_dims` must be less than or equal to tensor's rank.

    Args:
        tensor: A Tensor to flatten.
        preserve_dims (int): The number of leading dimensions to preserve.
        flattened_dim (int, optional): The size of the resulting flattened
            dimension. If not given, infer automatically.

    Returns:
        A Tensor with rank :attr:`preserve_dims` +1.

    Example:
        .. code-block:: python

            x = torch.ones(d_1, d_2, d_3, d_4)
            y = flatten(x, 2) # y.shape == [d_1, d_2, d_3 * d_4]
    """
    if preserve_dims > tensor.dim():
        raise ValueError(
            "`preserve_dims` must be less than or equal to tensor's rank")
    if flattened_dim is None:
        flattened_dim = -1
    shape = tensor.size()[:preserve_dims] + (flattened_dim,)
    tensor_ = tensor.reshape(shape)
    return tensor_


def pad_and_concat(values: List[torch.Tensor], axis: int,
                   pad_axis: Optional[MaybeList[int]] = None,
                   pad_constant_values: Any = 0) -> torch.Tensor:
    r"""Concatenates tensors along one dimension. Pads each of other dimensions
    of the tensors to the corresponding maximum size if necessary.

    Args:
        values: A list of Tensors of the same rank.
        axis (int): A Python int. Dimension along which to concatenate.
        pad_axis (int or list, optional): A Python int or a list of int.
            Dimensions to pad. Paddings are only added to the end of
            corresponding dimensions. If `None`, all dimensions except the
            :attr:`axis` dimension are padded.
        pad_constant_values: The scalar pad value to use. Must be same type
            as the tensors.

    Returns:
        A ``Tensor`` resulting from padding and concatenation of the input
        tensors.

    Raises:
        ValueError: If ``rank`` of :attr:`values` are not consistent.

    Example:

        .. code-block:: python

            a = torch.ones([1, 2])
            b = torch.ones([2, 3])

            c = pad_and_concat([a,b], 0)
            # c.shape == [3, 3]
            # c == [[1, 1, 0],
            #       [1, 1, 1],
            #       [1, 1, 1]]

            d = pad_and_concat([a,b], 1)
            # d.shape == [2, 5]
            # d == [[1, 1, 1, 1, 1]
            #       [0, 0, 1, 1, 1]]
    """
    rank = values[0].dim()
    if any(value.dim() != rank for value in values):
        raise ValueError("All tensors in `values` must have the same rank.")

    if pad_axis is None:
        pad_axis = [r for r in range(rank) if r != axis]
    elif isinstance(pad_axis, int):
        pad_axis = [pad_axis]
    for pad_dim in pad_axis:
        max_dim_size = max(v.size(pad_dim) for v in values)
        for i, v in enumerate(values):
            pad_shape: List[int] = list(v.size())
            if pad_shape[pad_dim] == max_dim_size:
                continue
            pad_shape[pad_dim] = max_dim_size - pad_shape[pad_dim]
            padding = values[0].new_full(tuple(pad_shape), pad_constant_values)
            values[i] = torch.cat((v, padding), dim=pad_dim)

    return torch.cat(values, dim=axis)
