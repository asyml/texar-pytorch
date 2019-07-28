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
Various entropies.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from texar.torch.losses.losses_utils import mask_and_reduce, reduce_dimensions
from texar.torch.utils.shapes import get_rank

__all__ = [
    "entropy_with_logits",
    "sequence_entropy_with_logits",
]


def _get_entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.

    Args:
        logits: Unscaled log probabilities.

    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, -1)
    return entropy


def entropy_with_logits(logits: torch.Tensor,
                        rank: Optional[int] = None,
                        average_across_batch: bool = True,
                        average_across_remaining: bool = False,
                        sum_over_batch: bool = False,
                        sum_over_remaining: bool = True) -> torch.Tensor:
    r"""Shannon entropy given logits.

    Args:
        logits: Unscaled log probabilities of shape
            `[batch_size, d_2, ..., d_{rank-1}, distribution_dim]`
            and of dtype `float32` or `float64`.

            The rank of the tensor is optionally specified by the argument
            :attr:`rank`.

            The tensor is considered as having `[batch_size, .., d_{rank-1}]`
            elements, each of which has a distribution of length `d_rank`
            (i.e., `distribution_dim`). So the last dimension is always
            summed out to compute the entropy.
        rank (int, optional): The rank of :attr:`logits`.
            If `None` (default), `rank` is inferred automatically from
            `logits`. If the inference fails, `rank` is
            set to 2, i.e., assuming :attr:`logits` is of shape
            `[batch_size, distribution_dim]`
        average_across_batch (bool): If set, average the entropy across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_remaining (bool): If set, average the entropy across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time.
            Used only when :attr:`logits` has rank >= 3.
        sum_over_batch (bool): If set, sum the entropy across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_remaining (bool): If set, sum the entropy across the
            remaining dimension. Must not set `average_across_remaining`
            and `sum_over_remaining` at the same time.
            Used only when :attr:`logits` has rank >= 3.

    Returns:
        A Tensor containing the Shannon entropy. The dimensionality of the
        Tensor depends on the configuration of reduction arguments. For
        example, if both batch and remaining dimensions are reduced (by
        either sum or average), the returned Tensor is a scalar Tensor.
    """
    entropy = _get_entropy(logits)

    if rank is None:
        rank = get_rank(logits)
    if rank is None:
        rank = 2
    rank -= 1

    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and "
                         "`sum_over_batch` can be set.")
    if average_across_remaining and sum_over_remaining:
        raise ValueError("Only one of `average_across_remaining` and "
                         "`sum_over_remaining` can be set.")
    sum_axes, average_axes = [], []
    if sum_over_batch:
        sum_axes.append(0)
    if average_across_batch:
        average_axes.append(0)
    if sum_over_remaining and rank >= 2:
        sum_axes += list(range(1, rank))
    if average_across_remaining and rank >= 2:
        average_axes += list(range(1, rank))

    entropy = reduce_dimensions(
        entropy, average_axes=average_axes, sum_axes=sum_axes
    )

    return entropy


def sequence_entropy_with_logits(
        logits: torch.Tensor,
        rank: Optional[int] = None,
        sequence_length: Optional[torch.LongTensor] = None,
        average_across_batch: bool = True,
        average_across_timesteps: bool = False,
        average_across_remaining: bool = False,
        sum_over_batch: bool = False,
        sum_over_timesteps: bool = True,
        sum_over_remaining: bool = True,
        time_major: bool = False) -> torch.Tensor:
    r"""Shannon entropy given logits.

    Args:
        logits: Unscaled log probabilities of shape
            `[batch_size, max_time, d_3, ..., d_{rank-1}, distribution_dim]`
            and of dtype `float32` or `float64`.

            The rank of the tensor is optionally specified by the argument
            :attr:`rank`.

            The tensor is considered as having `[batch_size, .., d_{rank-1}]`
            elements, each of which has a distribution of length `d_rank`
            (i.e., `distribution_dim`). So the last dimension is always
            summed out to compute the entropy.

            The batch and time dimensions are exchanged if :attr:`time_major`
            is `True`.
        rank (int, optional): The rank of :attr:`logits`.
            If `None` (default), `rank` is inferred automatically from
            `logits`. If the inference fails, `rank` is
            set to 3, i.e., assuming `logits` is of shape
            `[batch_size, max_time, distribution_dim]`
        sequence_length (optional): A Tensor of shape `[batch_size]`.
            Time steps beyond the respective sequence lengths are
            counted into the entropy.
        average_across_timesteps (bool): If set, average the entropy across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the entropy across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_remaining (bool): If set, average the entropy across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time.
            Used only when :attr:`logits` has rank >= 4.
        sum_over_timesteps (bool): If set, sum the entropy across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the entropy across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_remaining (bool): If set, sum the entropy across the
            remaining dimension. Must not set `average_across_remaining`
            and `sum_over_remaining` at the same time.
            Used only when :attr:`logits` has rank >= 4.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`logits` must have shape `[max_time, batch_size, ...]`.
            If `False` (default), it must have shape
            `[batch_size, max_time, ...]`.

    Returns:
        A Tensor containing the Shannon entropy. The dimensionality of the
        Tensor depends on the configuration of reduction arguments. For
        example, if batch, time, and remaining dimensions are all reduced (by
        either sum or average), the returned Tensor is a scalar Tensor.
    """
    entropy = _get_entropy(logits)

    if rank is None:
        rank = get_rank(logits)
    if rank is None:
        rank = 3
    rank -= 1

    entropy = mask_and_reduce(
        entropy,
        sequence_length,
        rank=rank,
        average_across_batch=average_across_batch,
        average_across_timesteps=average_across_timesteps,
        average_across_remaining=average_across_remaining,
        sum_over_batch=sum_over_batch,
        sum_over_timesteps=sum_over_timesteps,
        sum_over_remaining=sum_over_remaining,
        time_major=time_major
    )

    return entropy
