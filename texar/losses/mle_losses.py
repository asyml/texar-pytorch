# Copyright 2018 The Texar Authors. All Rights Reserved.
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
Various losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from texar.losses.losses_utils import mask_and_reduce, reduce_dimensions
from texar.utils import shapes

# pylint: disable=invalid-name, not-context-manager, protected-access,
# pylint: disable=too-many-arguments

__all__ = [
    "sequence_softmax_cross_entropy",
    "sequence_sparse_softmax_cross_entropy",
    "sequence_sigmoid_cross_entropy",
    "binary_sigmoid_cross_entropy",
    "binary_sigmoid_cross_entropy_with_clas"
]


def sequence_softmax_cross_entropy(labels,
                                   logits,
                                   sequence_length,
                                   average_across_batch=True,
                                   average_across_timesteps=False,
                                   sum_over_batch=False,
                                   sum_over_timesteps=True,
                                   time_major=False,
                                   stop_gradient_to_label=False,
                                   name=None):
    """Computes softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class distributions.

            - If :attr:`time_major` is `False` (default), this must be a\
            Tensor of shape `[batch_size, max_time, num_classes]`.

            - If `time_major` is `True`, this must be a Tensor of shape\
            `[max_time, batch_size, num_classes]`.

            Each row of `labels` should be a valid probability
            distribution, otherwise, the computation of the gradient will be
            incorrect.
        logits: Unscaled log probabilities. This must have the shape of
            `[max_time, batch_size, num_classes]` or
            `[batch_size, max_time, num_classes]` according to
            the value of `time_major`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will have zero losses.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`labels` and :attr:`logits` must have shape
            `[max_time, batch_size, ...]`. If `False`
            (default), they must have shape `[batch_size, max_time, ...]`.
        stop_gradient_to_label (bool): If set, gradient propagation to
            :attr:`labels` will be disabled.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
        For example:

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`  \
        are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are \
        `False`, the return Tensor is of shape `[max_time]`.
    """
    if stop_gradient_to_label:
        labels = labels.detach()

    losses = torch.sum(- labels * F.log_softmax(logits, -1), -1)

    losses = mask_and_reduce(losses,
                             sequence_length,
                             rank=2,
                             average_across_batch=average_across_batch,
                             average_across_timesteps=average_across_timesteps,
                             sum_over_batch=sum_over_batch,
                             sum_over_timesteps=sum_over_timesteps,
                             time_major=time_major)
    return losses


def sequence_sparse_softmax_cross_entropy(labels,
                                          logits,
                                          sequence_length,
                                          average_across_batch=True,
                                          average_across_timesteps=False,
                                          sum_over_batch=False,
                                          sum_over_timesteps=True,
                                          time_major=False,
                                          name=None):
    """Computes sparse softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class indexes. I.e., classes are mutually exclusive
            (each entry is in exactly one class).

            - If :attr:`time_major` is `False` (default), this must be\
            a Tensor of shape `[batch_size, max_time]`.

            - If `time_major` is `True`, this must be a Tensor of shape\
            `[max_time, batch_size].`
        logits: Unscaled log probabilities. This must have the shape of
            `[max_time, batch_size, num_classes]` or
            `[batch_size, max_time, num_classes]` according to
            the value of `time_major`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will have zero losses.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`labels` and :attr:`logits` must have shape
            `[max_time, batch_size, ...]`. If `False`
            (default), they must have shape `[batch_size, max_time, ...]`.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
        For example:

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`  \
        are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are \
        `False`, the return Tensor is of shape `[max_time]`.

    Example:

        .. code-block:: python

            embedder = WordEmbedder(vocab_size=data.vocab.size)
            decoder = BasicRNNDecoder(vocab_size=data.vocab.size)
            outputs, _, _ = decoder(
                decoding_strategy='train_greedy',
                inputs=embedder(data_batch['text_ids']),
                sequence_length=data_batch['length']-1)

            loss = sequence_sparse_softmax_cross_entropy(
                labels=data_batch['text_ids'][:, 1:],
                logits=outputs.logits,
                sequence_length=data_batch['length']-1)

    """
    losses = F.nll_loss(F.log_softmax(logits, dim=1), labels)

    losses = mask_and_reduce(losses,
                             sequence_length,
                             rank=2,
                             average_across_batch=average_across_batch,
                             average_across_timesteps=average_across_timesteps,
                             sum_over_batch=sum_over_batch,
                             sum_over_timesteps=sum_over_timesteps,
                             time_major=time_major)
    return losses


def sequence_sigmoid_cross_entropy(labels,
                                   logits,
                                   sequence_length,
                                   average_across_batch=True,
                                   average_across_timesteps=False,
                                   average_across_classes=True,
                                   sum_over_batch=False,
                                   sum_over_timesteps=True,
                                   sum_over_classes=False,
                                   time_major=False,
                                   stop_gradient_to_label=False,
                                   name=None):
    """Computes sigmoid cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class distributions.

            - If :attr:`time_major` is `False` (default), this must be a\
            Tensor of shape `[batch_size, max_time(, num_classes)]`.

            - If `time_major` is `True`, this must be a Tensor of shape\
            `[max_time, batch_size(, num_classes)]`.

            Each row of `labels` should be a valid probability
            distribution, otherwise, the computation of the gradient will be
            incorrect.
        logits: Unscaled log probabilities having the same shape as with
            :attr:`labels`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will have zero losses.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_classes (bool): If set, average the loss across the
            class dimension (if exists). Must not set
            `average_across_classes`' and `sum_over_classes` at
            the same time. Ignored if :attr:`logits` is a 2D Tensor.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_classes (bool): If set, sum the loss across the
            class dimension. Must not set `average_across_classes`
            and `sum_over_classes` at the same time. Ignored if
            :attr:`logits` is a 2D Tensor.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`labels` and :attr:`logits` must have shape
            `[max_time, batch_size, ...]`. If `False`
            (default), they must have shape `[batch_size, max_time, ...]`.
        stop_gradient_to_label (bool): If set, gradient propagation to
            :attr:`labels` will be disabled.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments
        :attr:`{average_across}/{sum_over}_{timesteps}/{batch}/{classes}`.
        For example, if the class dimension does not exist, and

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`  \
        are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are \
        `False`, the return Tensor is of shape `[max_time]`.
    """
    if stop_gradient_to_label:
        labels = labels.detach()




















