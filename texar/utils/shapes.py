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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# pylint: disable=no-name-in-module, protected-access, no-member, invalid-name

import torch
import numpy as np

__all__ = [
    "get_rank"
]

def get_rank(tensor):
    """Returns the tensor rank as a python `int`. The input tensor can also be
    a python array.

    Args:
        tensor: A Tensor or python array.

    Returns:
        A python `int` representing the rank of :attr:`tensor`. Returns
        `None` if the rank cannot be determined.
    """
    if isinstance(tensor, torch.Tensor):
        shape = tensor.shape
        try:
            rank = len(shape)
        except ValueError: # when `shape==TensorShape(None)`
            rank = None
    else:
        array = np.asarray(tensor)
        rank = array.ndim
    return rank

def mask_sequences(sequence,
                   sequence_length,
                   dtype=None,
                   time_major=False):
    """Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    This is the numpy version of :func:`texar.utils.mask_sequences`.

    Args:
        sequence: An python array of sequence values.

            If `time_major=False` (default), this must be an array of shape:
                `[batch_size, max_time, ...]`

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, ...].`
        sequence_length: An array of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            `[max_time, batch_size, ...]`.
            If `False` (default), :attr:`sequence` must have
            shape `[batch_size, max_time, ...]`.

    Returns:
        The masked sequence, i.e., an array of the same shape as
        :attr:`sequence` but with masked-out entries (set to zero).
    """
    tensor_date_type = sequence.dtype
    sequence = np.array(sequence)
    sequence_length = np.array(sequence_length)

    rank = sequence.ndim
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")
    batch_size = sequence.shape[0]
    max_time = sequence.shape[1]
    dtype = dtype or sequence.dtype

    if time_major:
        sequence = np.transpose(sequence, axes=[1, 0, 2])

    steps = np.tile(np.arange(max_time), [batch_size, 1])
    mask = np.asarray(steps < sequence_length[:, None], dtype=dtype)
    for _ in range(2, rank):
        mask = np.expand_dims(mask, -1)

    sequence = sequence * mask

    if time_major:
        sequence = np.transpose(sequence, axes=[1, 0, 2])
    sequence = torch.Tensor(sequence).type(tensor_date_type)
    return sequence