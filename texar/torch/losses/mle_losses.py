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
Various losses
"""

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from texar.torch.losses.losses_utils import mask_and_reduce, reduce_dimensions
from texar.torch.utils import shapes
from texar.torch.utils.types import MaybeTuple

__all__ = [
    "sequence_softmax_cross_entropy",
    "sequence_sparse_softmax_cross_entropy",
    "sequence_sigmoid_cross_entropy",
    "binary_sigmoid_cross_entropy",
    "binary_sigmoid_cross_entropy_with_clas",
]


def sequence_softmax_cross_entropy(
        labels: torch.Tensor,
        logits: torch.Tensor,
        sequence_length: Optional[torch.LongTensor],
        average_across_batch: bool = True,
        average_across_timesteps: bool = False,
        sum_over_batch: bool = False,
        sum_over_timesteps: bool = True,
        time_major: bool = False,
        stop_gradient_to_label: bool = False) -> torch.Tensor:
    r"""Computes softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class distributions.

            - If :attr:`time_major` is `False` (default), this must be a
              Tensor of shape `[batch_size, max_time, num_classes]`.

            - If `time_major` is `True`, this must be a Tensor of shape
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

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
        For example:

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`
          are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are
          `False`, the return Tensor is of shape `[max_time]`.
    """
    if stop_gradient_to_label:
        labels = labels.detach()

    losses = (-labels.type(logits.dtype) *
              F.log_softmax(logits, -1)).sum(dim=-1)

    losses = mask_and_reduce(losses,
                             sequence_length,
                             rank=2,
                             average_across_batch=average_across_batch,
                             average_across_timesteps=average_across_timesteps,
                             sum_over_batch=sum_over_batch,
                             sum_over_timesteps=sum_over_timesteps,
                             time_major=time_major)
    return losses


def sequence_sparse_softmax_cross_entropy(
        labels: torch.Tensor,
        logits: torch.Tensor,
        sequence_length: Optional[torch.LongTensor],
        average_across_batch: bool = True,
        average_across_timesteps: bool = False,
        sum_over_batch: bool = False,
        sum_over_timesteps: bool = True,
        time_major: bool = False) -> torch.Tensor:
    r"""Computes sparse softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class indexes. I.e., classes are mutually exclusive
            (each entry is in exactly one class).

            - If :attr:`time_major` is `False` (default), this must be
              a Tensor of shape `[batch_size, max_time]`.

            - If `time_major` is `True`, this must be a Tensor of shape
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

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
        For example:

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`
          are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are
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
    logits = F.log_softmax(logits, dim=2)
    logits = logits.permute(0, 2, 1)
    losses = F.nll_loss(logits, labels, reduction='none')

    losses = mask_and_reduce(losses,
                             sequence_length,
                             rank=2,
                             average_across_batch=average_across_batch,
                             average_across_timesteps=average_across_timesteps,
                             sum_over_batch=sum_over_batch,
                             sum_over_timesteps=sum_over_timesteps,
                             time_major=time_major)
    return losses


def sequence_sigmoid_cross_entropy(
        labels: torch.Tensor,
        logits: torch.Tensor,
        sequence_length: Optional[torch.LongTensor],
        average_across_batch: bool = True,
        average_across_timesteps: bool = False,
        average_across_classes: bool = True,
        sum_over_batch: bool = False,
        sum_over_timesteps: bool = True,
        sum_over_classes: bool = False,
        time_major: bool = False,
        stop_gradient_to_label: bool = False) -> torch.Tensor:
    r"""Computes sigmoid cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class distributions.

            - If :attr:`time_major` is `False` (default), this must be a
              Tensor of shape `[batch_size, max_time(, num_classes)]`.

            - If `time_major` is `True`, this must be a Tensor of shape
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

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments
        :attr:`{average_across}/{sum_over}_{timesteps}/{batch}/{classes}`.
        For example, if the class dimension does not exist, and

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`
          are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are
          `False`, the return Tensor is of shape `[max_time]`.
    """
    if stop_gradient_to_label:
        labels = labels.detach()
    losses = F.binary_cross_entropy_with_logits(
        logits, labels.type(logits.dtype), reduction='none')

    rank = shapes.get_rank(logits) or shapes.get_rank(labels)

    losses = mask_and_reduce(losses,
                             sequence_length,
                             rank=rank,
                             average_across_batch=average_across_batch,
                             average_across_timesteps=average_across_timesteps,
                             average_across_remaining=average_across_classes,
                             sum_over_batch=sum_over_batch,
                             sum_over_timesteps=sum_over_timesteps,
                             sum_over_remaining=sum_over_classes,
                             time_major=time_major)

    return losses


def binary_sigmoid_cross_entropy(
        pos_logits: Optional[torch.Tensor] = None,
        neg_logits: Optional[torch.Tensor] = None,
        average_across_batch: bool = True,
        average_across_classes: bool = True,
        sum_over_batch: bool = False,
        sum_over_classes: bool = False,
        return_pos_neg_losses: bool = False) \
        -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""Computes sigmoid cross entropy of binary predictions.

    Args:
        pos_logits: The logits of predicting positive on positive data. A
            tensor of shape `[batch_size(, num_classes)]`.
        neg_logits: The logits of predicting positive on negative data. A
            tensor of shape `[batch_size(, num_classes)]`.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_classes (bool): If set, average the loss across the
            class dimension (if exists). Must not set
            `average_across_classes`' and `sum_over_classes` at
            the same time. Ignored if :attr:`logits` is a 1D Tensor.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_classes (bool): If set, sum the loss across the
            class dimension. Must not set `average_across_classes`
            and `sum_over_classes` at the same time. Ignored if
            :attr:`logits` is a 2D Tensor.
        return_pos_neg_losses (bool): If set, additionally returns the losses
            on :attr:`pos_logits` and :attr:`neg_logits`, respectively.

    Returns:
        By default, a Tensor containing the loss, of rank 0, 1, or 2 depending
        on the arguments :attr:`{average_across}/{sum_over}_{batch}/{classes}`.
        For example:

            - If :attr:`sum_over_batch` and :attr:`average_across_classes`
              are `True` (default), the return Tensor is of rank 0.

            - If  arguments are `False`, the return Tensor is of shape
              `[batch_size(, num_classes)]`.

        If :attr:`return_pos_neg_losses` is `True`, returns a tuple
        `(loss, pos_loss, neg_loss)`, where `loss` is the loss above;
        `pos_loss` is the loss on `pos_logits` only; and
        `neg_loss` is the loss on `neg_logits` only. They have
        `loss = pos_loss + neg_loss`.
    """
    average_axes = [0] if average_across_batch else []
    average_axes += [1] if average_across_classes else []
    sum_axes = [0] if sum_over_batch else []
    sum_axes += [1] if sum_over_classes else []

    if pos_logits is not None:
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction='none')

        pos_loss = reduce_dimensions(pos_loss, average_axes, sum_axes)
    else:
        pos_loss = 0  # type: ignore

    if neg_logits is not None:
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction='none')

        neg_loss = reduce_dimensions(neg_loss, average_axes, sum_axes)
    else:
        neg_loss = 0  # type: ignore

    loss = pos_loss + neg_loss

    if return_pos_neg_losses:
        return loss, pos_loss, neg_loss
    else:
        return loss


def binary_sigmoid_cross_entropy_with_clas(
        clas_fn: Callable[[torch.Tensor], MaybeTuple[torch.Tensor]],
        pos_inputs: Optional[torch.Tensor] = None,
        neg_inputs: Optional[torch.Tensor] = None,
        average_across_batch: bool = True,
        average_across_classes: bool = True,
        sum_over_batch: bool = False,
        sum_over_classes: bool = False,
        return_pos_neg_losses: bool = False) \
        -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""Computes sigmoid cross entropy of binary classifier.

    Args:
        clas_fn: A callable takes data (e.g., :attr:`pos_inputs` and
            :attr:`fake_inputs`) and returns the logits of being positive. The
            signature of `clas_fn` must be:
            :python:`logits (, ...) = clas_fn(inputs)`.
            The return value of `clas_fn` can be the logits, or
            a tuple where the logits are the first element.
        pos_inputs: The positive data fed into `clas_fn`.
        neg_inputs: The negative data fed into `clas_fn`.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_classes (bool): If set, average the loss across the
            class dimension (if exists). Must not set
            `average_across_classes`' and `sum_over_classes` at
            the same time. Ignored if :attr:`logits` is a 1D Tensor.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_classes (bool): If set, sum the loss across the
            class dimension. Must not set `average_across_classes`
            and `sum_over_classes` at the same time. Ignored if
            :attr:`logits` is a 2D Tensor.
        return_pos_neg_losses (bool): If set, additionally returns the losses
            on :attr:`pos_logits` and :attr:`neg_logits`, respectively.

    Returns:
        By default, a Tensor containing the loss, of rank 0, 1, or 2 depending
        on the arguments :attr:`{average_across}/{sum_over}_{batch}/{classes}`.
        For example:

            - If :attr:`sum_over_batch` and :attr:`average_across_classes`
              are `True` (default), the return Tensor is of rank 0.

            - If  arguments are `False`, the return Tensor is of shape
              `[batch_size(, num_classes)]`.

        If :attr:`return_pos_neg_losses`=`True`, returns a tuple
        `(loss, pos_loss, neg_loss)`, where `loss` is the loss above;
        `pos_loss` is the loss on `pos_logits` only; and
        `neg_loss` is the loss on `neg_logits` only. They have
        `loss = pos_loss + neg_loss`.
    """
    pos_logits = None
    if pos_inputs is not None:
        out = clas_fn(pos_inputs)
        pos_logits = out[0] if isinstance(out, (list, tuple)) else out

    neg_logits = None
    if neg_inputs is not None:
        out = clas_fn(neg_inputs)
        neg_logits = out[0] if isinstance(out, (list, tuple)) else out

    return binary_sigmoid_cross_entropy(
        pos_logits=pos_logits,
        neg_logits=neg_logits,
        average_across_batch=average_across_batch,
        average_across_classes=average_across_classes,
        sum_over_batch=sum_over_batch,
        sum_over_classes=sum_over_classes,
        return_pos_neg_losses=return_pos_neg_losses)
