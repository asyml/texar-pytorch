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
Various reward related functions.
"""

from typing import Optional

import torch

from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.utils import sequence_mask

__all__ = [
    "discount_reward",
    "_discount_reward_tensor_1d",
    "_discount_reward_tensor_2d",
]


def discount_reward(reward: torch.Tensor,
                    sequence_length: Optional[torch.LongTensor] = None,
                    discount: float = 1.,
                    normalize: bool = False) -> torch.Tensor:
    r"""Computes discounted reward.

    Args:
        reward: A Tensor. Can be 1D with shape `[batch_size]`,
            or 2D with shape `[batch_size, max_time]`.
        sequence_length (optional): A Tensor of shape `[batch_size]`.
            Time steps beyond the respective sequence lengths will be masked.
            Required if :attr:`reward` is 1D.
        discount (float): A scalar. The discount factor.
        normalize (bool): Whether to normalize the discounted reward, by
            `(discounted_reward - mean) / std`. Here `mean` and `std` are
            over all time steps and all samples in the batch.

    Returns:
        A 2D Tensor of the discounted reward.
    """
    if not isinstance(reward, torch.Tensor):
        reward = torch.tensor(reward)
    if (sequence_length is not None and
            not isinstance(sequence_length, torch.Tensor)):
        sequence_length = torch.tensor(
            sequence_length, dtype=torch.int64, device=reward.device)

    tensor_rank = reward.dim()
    if tensor_rank == 1:
        disc_reward = _discount_reward_tensor_1d(  # type: ignore
            reward, sequence_length, discount)
    elif tensor_rank == 2:
        disc_reward = _discount_reward_tensor_2d(
            reward, sequence_length, discount)
    else:
        raise ValueError("The dimension of reward can only be 1 or 2.")

    if normalize:
        mu = torch.mean(disc_reward)
        var = torch.std(disc_reward)
        disc_reward = (disc_reward - mu) / (torch.sqrt(var) + 1e-8)

    return disc_reward


def _discount_reward_tensor_1d(reward: torch.Tensor,
                               sequence_length: torch.LongTensor,
                               discount: float = 1.) -> torch.Tensor:
    r"""Computes discounted reward.

    Args:
        reward: 1D Tensor with shape `[batch_size]`.
        sequence_length: A Tensor of shape `[batch_size]`.
        Time steps beyond the respective sequence lengths will be masked.
        discount (float): A scalar. The discount factor.

    Returns:
        A 2D Tensor of the discounted reward.
    """
    if sequence_length is None:
        raise ValueError('sequence_length must not be `None` for 1D reward.')

    if not isinstance(sequence_length, torch.Tensor):
        sequence_length = torch.tensor(
            sequence_length, dtype=torch.int64, device=reward.device)

    batch_size = reward.shape[0]
    max_seq_length = torch.max(sequence_length)
    dtype: torch.dtype = reward.dtype

    if discount == 1.:
        disc_reward = reward.unsqueeze(-1).expand(batch_size, max_seq_length)
    else:
        mask = sequence_mask(sequence_length, dtype=dtype)
        mask = torch.cat((mask[:, 1:], torch.zeros_like(mask[:, -1:])), dim=1)
        # Make each row = [discount, ..., discount, 1, ..., 1]
        dmat = mask * discount + (1 - mask)
        dmat = torch.flip(dmat, (1,))
        dmat = torch.cumprod(dmat, dim=1)
        dmat = torch.flip(dmat, (1,))
        disc_reward = dmat * reward.unsqueeze(-1)

    disc_reward = mask_sequences(disc_reward, sequence_length, dtype=dtype)

    return disc_reward


def _discount_reward_tensor_2d(
        reward: torch.Tensor,
        sequence_length: Optional[torch.LongTensor] = None,
        discount: float = 1.) -> torch.Tensor:
    r"""Computes discounted reward.

    Args:
        reward: 2D Tensor with shape `[batch_size, max_time]`.
        sequence_length (optional): A Tensor of shape `[batch_size]`.
            Time steps beyond the respective sequence lengths will be masked.
        discount (float): A scalar. The discount factor.

    Returns:
        A 2D Tensor of the discounted reward.
    """
    dtype: torch.dtype = reward.dtype
    if sequence_length is not None:
        reward = mask_sequences(reward, sequence_length, dtype=dtype)

    if discount == 1.:
        reward = torch.flip(reward, (1,))
        disc_reward = torch.cumsum(reward, dim=1)
        disc_reward = torch.flip(disc_reward, (1,))
    else:
        # [max_time, batch_size]
        rev_reward_T = torch.flip(reward, (1,)).permute(1, 0)

        res = []
        acc = torch.zeros_like(reward[:, 1])
        for i in range(rev_reward_T.shape[0]):
            cur = rev_reward_T[i]
            acc = cur + discount * acc
            res.append(acc)

        rev_reward_T_cum = torch.stack(res, dim=0)
        disc_reward = torch.flip(rev_reward_T_cum.permute(1, 0), (1,))

    return disc_reward
