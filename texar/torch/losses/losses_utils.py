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
Various utilities for losses.
"""

from typing import Optional

import torch

from texar.torch.utils.shapes import mask_sequences, transpose_batch_time
from texar.torch.utils.types import MaybeList

__all__ = [
    "mask_and_reduce",
    "reduce_batch_time",
    "reduce_dimensions",
]


def mask_and_reduce(sequence: torch.Tensor,
                    sequence_length: Optional[torch.LongTensor],
                    rank: int = 2,
                    average_across_batch: bool = True,
                    average_across_timesteps: bool = False,
                    average_across_remaining: bool = False,
                    sum_over_batch: bool = False,
                    sum_over_timesteps: bool = True,
                    sum_over_remaining: bool = True,
                    dtype: Optional[torch.dtype] = None,
                    time_major: bool = False) -> torch.Tensor:
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths, and reduces (average or sum) away dimensions.

    This is a combination of :func:`~texar.torch.utils.shapes.mask_sequences`
    and :func:`~texar.torch.losses.losses_utils.reduce_batch_time`.

    Args:
        sequence: A tensor of sequence values.
            If `time_major=False` (default), this must be a tensor of shape
            `[batch_size, max_time, d_2, ..., d_rank]`, where the rank of
            the tensor is specified with :attr:`rank`.
            The batch and time dimensions are exchanged if `time_major` is True.
        sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero. If `None`,
            no masking is performed.
        rank (int): The rank of :attr:`sequence`. Must be >= 2. Default is 2,
            i.e., `sequence` is a 2D Tensor consisting of batch and time
            dimensions.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_remaining (bool): If set, average the sequence across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time.
        sum_over_timesteps (bool): If set, sum the sequence across the time
            dimension. Must not set `average_across_timesteps` and
            `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the sequence across the batch
            dimension. Must not set `average_across_batch` and `sum_over_batch`
            at the same time.
        sum_over_remaining (bool): If set, sum the sequence across the remaining
            dimension. Must not set `average_across_remaining` and
            `sum_over_remaining` at the same time.
        dtype (torch.dtype): The dtype of the returned mask.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape `[max_time, batch_size, ...]`.
            If `False` (default), `sequence` must have
            shape `[batch_size, max_time, ...]`.

    Returns:
        A tensor containing the masked and reduced sequence.
    """
    if rank < 2:
        raise ValueError('`rank` must be >= 2.')

    if time_major:
        sequence = transpose_batch_time(sequence)

    if sequence_length is not None:
        sequence = mask_sequences(sequence,
                                  sequence_length,
                                  dtype=dtype,
                                  time_major=False)
    if rank > 2:
        if average_across_remaining and sum_over_remaining:
            raise ValueError("Only one of `average_across_remaining` and "
                             "`sum_over_remaining` can be set.")
        if average_across_remaining:
            for axis in sorted(list(range(2, rank)), reverse=True):
                sequence = torch.mean(sequence, dim=axis)
        elif sum_over_remaining:
            for axis in sorted(list(range(2, rank)), reverse=True):
                sequence = torch.sum(sequence, dim=axis)

    sequence = reduce_batch_time(sequence,
                                 sequence_length,
                                 average_across_batch,
                                 average_across_timesteps,
                                 sum_over_batch,
                                 sum_over_timesteps)

    reduce_time = average_across_timesteps or sum_over_timesteps
    reduce_batch = average_across_batch or sum_over_batch
    if not reduce_time and not reduce_batch and time_major:
        sequence = transpose_batch_time(sequence)

    return sequence


def reduce_batch_time(sequence: torch.Tensor,
                      sequence_length: Optional[torch.LongTensor],
                      average_across_batch: bool = True,
                      average_across_timesteps: bool = False,
                      sum_over_batch: bool = False,
                      sum_over_timesteps: bool = True) -> torch.Tensor:
    r"""Average or sum over the respective dimensions of :attr:`sequence`, which
    is of shape `[batch_size, max_time, ...]`.

    Assumes :attr:`sequence` has been properly masked according to
    :attr:`sequence_length`.

    Args:
        sequence: A tensor to reduce.
        sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero. If `None`,
            no masking is performed.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the sequence across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the sequence across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.

    Returns:
        A tensor with dimension reduction.
    """
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError("Only one of `average_across_timesteps` and "
                         "`sum_over_timesteps` can be set.")
    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and "
                         "`sum_over_batch` can be set.")

    if sum_over_timesteps:
        sequence = torch.sum(sequence, dim=1)
    elif average_across_timesteps:
        if sequence_length is None:
            sequence = torch.mean(sequence, dim=1)
        else:
            sequence = (torch.sum(sequence, dim=1).float() /
                        sequence_length.float())

    if sum_over_batch:
        sequence = torch.sum(sequence, dim=0)
    elif average_across_batch:
        sequence = torch.mean(sequence, dim=0)

    return sequence


def reduce_dimensions(tensor: torch.Tensor,
                      average_axes: Optional[MaybeList[int]] = None,
                      sum_axes: Optional[MaybeList[int]] = None,
                      keepdims: Optional[bool] = None) -> torch.Tensor:
    r"""Average or sum over dimensions of :attr:`tensor`.

    :attr:`average_axes` and :attr:`sum_axes` must be mutually exclusive. That
    is, elements in `average_axes` must not be contained in
    `sum_axes`, and vice versa.

    Args:
        tensor: A tensor to reduce.
        average_axes (optional): A (list of) `int` that indicates the
            dimensions to reduce by taking average.
        sum_axes (optional): A (list of) `int` that indicates the
            dimensions to reduce by taking sum.
        keepdims (optional): If `True`, retains reduced dimensions with
            length 1.

    Returns:
        A tensor with dimension reduction.
    """
    reduced_axes = set()
    if average_axes is not None:
        if not isinstance(average_axes, (list, tuple)):
            average_axes = [average_axes]
        if len(average_axes) > 0:
            for average_axis in average_axes:
                tensor = torch.mean(tensor, dim=average_axis, keepdim=True)
            reduced_axes.update(average_axes)

    if sum_axes is not None:
        if not isinstance(sum_axes, (list, tuple)):
            sum_axes = [sum_axes]
        if len(sum_axes) > 0:
            for sum_axis in sum_axes:
                tensor = torch.sum(tensor, dim=sum_axis, keepdim=True)
            reduced_axes.update(sum_axes)

            if average_axes is not None:
                if len(reduced_axes) != len(average_axes) + len(sum_axes):
                    raise ValueError('`average_axes` and `sum_axes` must not '
                                     'have overlapped elements.')
    if not keepdims:
        for axis in sorted(list(reduced_axes), reverse=True):
            tensor = torch.squeeze(tensor, dim=axis)
    return tensor
