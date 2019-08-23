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
Various helper classes and utilities for attention cell wrappers.

The code structure adapted from:
    `https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/
    seq2seq/python/ops/attention_wrapper.py`
"""

import functools
from abc import ABC
from typing import Callable, List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from texar.torch.core.attention_mechanism_utils import (
    maybe_mask_score, prepare_memory, safe_cumprod)
from texar.torch.module_base import ModuleBase
from texar.torch.utils.types import MaybeList, MaybeTuple

__all__ = [
    "AttentionMechanism",
    "AttentionWrapperState",
    "LuongAttention",
    "BahdanauAttention",
    "compute_attention",
    "monotonic_attention",
    "BahdanauMonotonicAttention",
    "LuongMonotonicAttention",
]

State = TypeVar('State')


class AttentionMechanism(ModuleBase, ABC):
    r"""A base AttentionMechanism class providing common functionality.

    Common functionality includes:

    1. Storing the query and memory layers.
    2. Preparing the score mask value.

    Args:
        encoder_output_size: The output size of the encoder cell.
        memory_layer: Instance of `torch.nn.Linear`. The layer's depth must
            match the depth of ``query_layer``.
        query_layer (optional): Instance of `torch.nn.Linear`. The layer's
            depth must  match the depth of ``memory_layer``.  If
            ``query_layer`` is not provided, the shape of ``query`` must
            match that of ``memory_layer``.
        score_mask_value (optional): The mask value for score before
            passing into `probability_fn`. The default is -inf. Only used
            if `memory_sequence_length` is not None.
    """

    # Cached variables that are initialized by transforming the `memory` at
    # the first forward pass of each batch. `clear_cache` should be called when
    # the batch is finished to prevent holding references to variables in the
    # computation graph.
    _values: torch.Tensor
    _keys: torch.Tensor

    def __init__(self,
                 encoder_output_size: int,
                 memory_layer: nn.Module,
                 query_layer: Optional[nn.Module] = None,
                 score_mask_value: Optional[torch.Tensor] = None):
        super().__init__()

        if (query_layer is not None and
                not isinstance(query_layer, nn.Linear)):
            raise TypeError("query_layer is not a Linear Layer: %s"
                            % type(query_layer).__name__)
        if (memory_layer is not None and
                not isinstance(memory_layer, nn.Linear)):
            raise TypeError("memory_layer is not a Linear Layer: %s"
                            % type(memory_layer).__name__)
        self._query_layer = query_layer
        self._memory_layer = memory_layer

        if score_mask_value is None:
            score_mask_value = torch.tensor(-np.inf)
        self.score_mask_value = score_mask_value

        self._encoder_output_size = encoder_output_size

        self._values = None  # type: ignore
        self._keys = None  # type: ignore

    def _process_query_and_memory(self, query: torch.Tensor,
                                  memory: torch.Tensor,
                                  memory_sequence_length: Optional[
                                      torch.Tensor] = None) -> torch.Tensor:
        r"""Preprocess the memory and query.

        Args:
            query: tensor, shaped ``[batch_size, query_depth]``.
            memory: the memory to query; usually the output of an RNN encoder.
                This tensor should be shaped ``[batch_size, max_time, ...]``.
            memory_sequence_length (optional): sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.
        """
        query = self._query_layer(query) if self._query_layer else query

        if self._values is None and self._keys is None:
            self._values = prepare_memory(memory, memory_sequence_length)

            if self._memory_layer is not None:
                self._keys = self._memory_layer(self._values)
            else:
                self._keys = self._values
        return query

    def forward(self,  # type: ignore
                query: torch.Tensor,
                state: torch.Tensor,
                memory: torch.Tensor,
                memory_sequence_length: Optional[torch.LongTensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Score the query based on the keys and values.

        Args:
            query: tensor, shaped ``[batch_size, query_depth]``.
            state: tensor, shaped ``[batch_size, alignments_size]``
                (``alignments_size`` is memory's ``max_time``).
            memory: the memory to query; usually the output of an RNN encoder.
                This tensor should be shaped ``[batch_size, max_time, ...]``.
            memory_sequence_length (optional): sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.

        Returns:
            Tensor of dtype matching ``memory`` and shape
            ``[batch_size, alignments_size]`` (``alignments_size`` is memory's
            ``max_time``).
        """
        raise NotImplementedError

    @property
    def memory_layer(self) -> nn.Module:
        r"""The layer used to transform the attention memory."""
        return self._memory_layer

    @property
    def query_layer(self) -> Optional[nn.Module]:
        r"""The layer used to transform the attention query."""
        return self._query_layer

    @property
    def values(self) -> torch.Tensor:
        r"""Cached tensor of the attention values."""
        return self._values

    @property
    def encoder_output_size(self) -> int:
        r"""Dimension of the encoder output."""
        return self._encoder_output_size

    def clear_cache(self):
        r"""Clear the cached preprocessed ``memory`` in the attention mechanism.
        This function should be called at the end of `forward()` in
        `AttentionRNNDecoder`.
        """
        self._values = None
        self._keys = None

    def initial_alignments(self,
                           batch_size: int,
                           max_time: int,
                           dtype: torch.dtype,
                           device: torch.device) -> torch.Tensor:
        r"""Creates the initial alignment values for the ``AttentionWrapper``
        class.

        This is important for ``AttentionMechanisms`` that use the previous
        alignment to calculate the alignment at the next time step
        (e.g. monotonic attention).

        The default behavior is to return a tensor of all zeros.

        Args:
            batch_size: integer scalar, the batch_size.
            max_time: integer scalar, the max_time (length of the source
                sequence).
            dtype: The `torch.dtype`.
            device: The `torch.device`.

        Returns:
            A ``dtype`` tensor shaped ``[batch_size, alignments_size]``
            (``alignments_size`` is the value of ``max_time``).
        """
        return torch.zeros(batch_size, max_time, dtype=dtype, device=device)

    def initial_state(self,
                      batch_size: int,
                      max_time: int,
                      dtype: torch.dtype,
                      device: torch.device) -> torch.Tensor:
        r"""Creates the initial state values for the ``AttentionWrapper`` class.

        This is important for ``AttentionMechanisms`` that use the previous
        alignment to calculate the alignment at the next time step
        (e.g. monotonic attention).

        The default behavior is to return the same output as
        ``initial_alignments``.

        Args:
            batch_size: integer scalar, the batch_size.
            max_time: integer scalar, the max_time (length of the source
                sequence).
            dtype: The `torch.dtype`.
            device: The `torch.device`.

        Returns:
            A ``dtype`` tensor shaped ``[batch_size, alignments_size]``
            (``alignments_size`` is the value of ``max_time``).
        """
        return self.initial_alignments(batch_size, max_time, dtype, device)


def _luong_score(query: torch.Tensor,
                 keys: torch.Tensor,
                 scale: Optional[torch.Tensor]) -> torch.Tensor:
    r"""Implements Luong-style (multiplicative) scoring function.
    This attention has two forms.

    The first is standard Luong attention, as described in:
    `Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.  https://arxiv.org/abs/1508.04025`_

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.

    To enable the second form, call this function with `scale=True`.

    Args:
        query: tensor, shape ``[batch_size, num_units]`` to compare to keys.
        keys: processed memory, shape ``[batch_size, max_time, num_units]``.
        scale (optional): tensor to scale the attention score.

    Returns:
        A ``[batch_size, max_time]`` tensor of unnormalized score values.

    Raises:
        ValueError: If ``key`` and ``query`` depths do not match.
    """
    depth = query.shape[-1]
    key_units = keys.shape[-1]
    if depth != key_units:
        raise ValueError(
            "Incompatible or unknown inner dimensions between query and keys. "
            "Query (%s) has units: %s.  Keys (%s) have units: %s. "
            "Perhaps you need to set num_units to the keys' dimension (%s)?" %
            (query, depth, keys, key_units, key_units))

    # Reshape from [batch_size, depth] to [batch_size, 1, depth] for matmul.
    query = torch.unsqueeze(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                keys is [batch_size, max_time, depth].
    # the inner product is asked to transpose keys' inner shape to get a batched
    #  matmul on: [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of: [batch_size, 1, max_time].
    # we then squeeze out the center singleton dimension.
    score = torch.matmul(query, keys.permute(0, 2, 1))
    score = torch.squeeze(score, 1)

    if scale is not None:
        # Scalar used in weight scaling
        score = scale * score
    return score


class LuongAttention(AttentionMechanism):
    r"""Implements Luong-style (multiplicative) attention scoring. This
    attention has two forms.

    The first is standard Luong attention, as described in:
    `Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    [Effective Approaches to Attention-based Neural Machine Translation.
    EMNLP 2015.] <https://arxiv.org/abs/1508.04025>`_

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention. To enable the second form, construct the object with
    parameter `scale=True`.

    Args:
        num_units: The depth of the attention mechanism.
        encoder_output_size: The output size of the encoder cell.
        scale: Python boolean.  Whether to scale the energy term.
        probability_fn (optional) A `callable`.  Converts the score to
            probabilities.  The default is `torch.nn.softmax`. Other options
            include :func:`~texar.torch.core.hardmax` and
            :func:`~texar.torch.core.sparsemax`. Its signature should be:
            :python:`probabilities = probability_fn(score)`.
        score_mask_value (optional) The mask value for score before passing
            into `probability_fn`. The default is `-inf`. Only used if
            :attr:`memory_sequence_length` is not None.
    """

    def __init__(self,
                 num_units: int,
                 encoder_output_size: int,
                 scale: bool = False,
                 probability_fn: Optional[Callable[[torch.Tensor],
                                                   torch.Tensor]] = None,
                 score_mask_value: Optional[torch.Tensor] = None):
        # For LuongAttention, we only transform the memory layer; thus
        # num_units must match expected the query depth.
        if probability_fn is None:
            probability_fn = lambda x: F.softmax(x, dim=-1)
        self._probability_fn = probability_fn

        super().__init__(
            encoder_output_size=encoder_output_size,
            memory_layer=nn.Linear(encoder_output_size, num_units, False),
            query_layer=None,
            score_mask_value=score_mask_value)

        self.attention_g: Optional[torch.Tensor] = None
        if scale:
            self.attention_g = nn.Parameter(torch.tensor(1.0),
                                            requires_grad=True)

    def forward(self,  # type: ignore
                query: torch.Tensor,
                state: torch.Tensor,
                memory: torch.Tensor,
                memory_sequence_length: Optional[torch.LongTensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        query = self._process_query_and_memory(
            query, memory, memory_sequence_length)

        score = _luong_score(query, self._keys, self.attention_g)

        alignments = self._probability_fn(
            maybe_mask_score(score, self.score_mask_value,
                             memory_sequence_length))

        next_state = alignments
        return alignments, next_state


def _bahdanau_score(processed_query: torch.Tensor,
                    keys: torch.Tensor,
                    attention_v: torch.Tensor,
                    attention_g: Optional[torch.Tensor] = None,
                    attention_b: Optional[torch.Tensor] = None):
    r"""Implements Bahdanau-style (additive) scoring function.
    This attention has two forms.

    The first is Bhandanau attention, as described in:
    `Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. <https://arxiv.org/abs/1409.0473>`_

    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    `Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks." <https://arxiv.org/abs/1602.07868>`_
    To enable the second form, set please pass in attention_g and attention_b.

    Args:
        processed_query: Tensor, shape ``[batch_size, num_units]`` to compare to
            keys.
        keys: Processed memory, shape ``[batch_size, max_time, num_units]``.
        attention_v: Tensor, shape ``[num_units]``.
        attention_g: Optional scalar tensor for normalization.
        attention_b: Optional tensor with shape ``[num_units]`` for
            normalization.

    Returns:
        A ``[batch_size, max_time]`` tensor of unnormalized score values.
    """
    processed_query = torch.unsqueeze(processed_query, 1)
    if attention_g is not None and attention_b is not None:
        normed_v = attention_g * attention_v * torch.rsqrt(
            torch.sum(attention_v ** 2))
        return torch.sum(normed_v * torch.tanh(keys + processed_query
                                               + attention_b), 2)
    else:
        return torch.sum(attention_v * torch.tanh(keys + processed_query), 2)


class BahdanauAttention(AttentionMechanism):
    r"""Implements Bahdanau-style (additive) attention.
    This attention has two forms.

    The first is Bahdanau attention, as described in:
    `Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. <https://arxiv.org/abs/1409.0473>`_

    The second is the normalized form. This form is inspired by the
    weight normalization article:
    `Tim Salimans, Diederik P. Kingma. "Weight Normalization: A Simple
    Reparameterization to Accelerate Training of Deep Neural Networks."
    <https://arxiv.org/abs/1602.07868>`_
    To enable the second form, construct the object with parameter
    `normalize=True`.

    Args:
        num_units: The depth of the query mechanism.
        decoder_output_size: The output size of the decoder cell.
        encoder_output_size: The output size of the encoder cell.
        normalize: bool.  Whether to normalize the energy term.
        probability_fn (optional) A `callable`.  Converts the score to
            probabilities.  The default is `torch.nn.softmax`. Other options
            include :func:`~texar.torch.core.hardmax` and
            :func:`~texar.torch.core.sparsemax`. Its signature should be:
            :python:`probabilities = probability_fn(score)`:.
        score_mask_value (optional): The mask value for score before passing
            into ``probability_fn``. The default is `-inf`. Only used
            if :attr:`memory_sequence_length` is not None.
    """

    def __init__(self,
                 num_units: int,
                 decoder_output_size: int,
                 encoder_output_size: int,
                 normalize: bool = False,
                 probability_fn: Optional[Callable[[torch.Tensor],
                                                   torch.Tensor]] = None,
                 score_mask_value: Optional[torch.Tensor] = None):
        if probability_fn is None:
            probability_fn = lambda x: F.softmax(x, dim=-1)
        self._probability_fn = probability_fn

        super().__init__(
            encoder_output_size=encoder_output_size,
            query_layer=nn.Linear(decoder_output_size, num_units, False),
            memory_layer=nn.Linear(encoder_output_size, num_units, False),
            score_mask_value=score_mask_value)

        limit = np.sqrt(3. / num_units)
        self.attention_v = 2 * limit * torch.rand(num_units) - limit
        self.attention_v = nn.Parameter(self.attention_v,
                                        requires_grad=True)

        self.attention_g: Optional[torch.Tensor]
        self.attention_b: Optional[torch.Tensor]
        if normalize:
            self.attention_g = torch.sqrt(torch.tensor(1. / num_units))
            self.attention_g = nn.Parameter(self.attention_g,
                                            requires_grad=True)
            self.attention_b = torch.zeros(num_units)
            self.attention_b = nn.Parameter(self.attention_b,
                                            requires_grad=True)
        else:
            self.attention_g = None
            self.attention_b = None

    def forward(self,  # type: ignore
                query: torch.Tensor,
                state: torch.Tensor,
                memory: torch.Tensor,
                memory_sequence_length: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        query = self._process_query_and_memory(
            query, memory, memory_sequence_length)

        score = _bahdanau_score(query,
                                self._keys,
                                self.attention_v,
                                self.attention_g,
                                self.attention_b)

        alignments = self._probability_fn(
            maybe_mask_score(score, self.score_mask_value,
                             memory_sequence_length))

        next_state = alignments
        return alignments, next_state


def monotonic_attention(p_choose_i: torch.Tensor,
                        previous_attention: torch.Tensor,
                        mode: str) -> torch.Tensor:
    r"""Compute monotonic attention distribution from choosing probabilities.
    Monotonic attention implies that the input sequence is processed in an
    explicitly left-to-right manner when generating the output sequence.  In
    addition, once an input sequence element is attended to at a given output
    time step, elements occurring before it cannot be attended to at subsequent
    output time steps.  This function generates attention distributions
    according to these assumptions. For more information, see `Online and
    Linear-Time Attention by Enforcing Monotonic Alignments`.

    Args:
        p_choose_i: Probability of choosing input sequence/memory element i.
            Should be of shape (batch_size, input_sequence_length), and should
            all be in the range [0, 1].
        previous_attention: The attention distribution from the previous output
            time step.  Should be of shape (batch_size, input_sequence_length).
            For the first output time step, `previous_attention[n]` should be
            `[1, 0, 0, ..., 0] for all n in [0, ... batch_size - 1]`.
        mode: How to compute the attention distribution.
            Must be one of ``"recursive"``, ``"parallel"``, or ``"hard"``:

            - ``"recursive"`` recursively computes the distribution.
              This is slowest but is exact, general, and does not suffer
              from numerical instabilities.
            - ``"parallel"`` uses parallelized cumulative-sum and
              cumulative-product operations to compute a closed-form
              solution to the recurrence relation defining the attention
              distribution. This makes it more efficient than
              ``"recursive"``, but it requires numerical checks which make
              the distribution non-exact. This can be a problem in
              particular when input sequence is long and/or
              :attr:`p_choose_i` has entries very close to 0 or 1.
            - ``"hard"`` requires that the probabilities in
              :attr:`p_choose_i` are all either 0 or 1, and subsequently
              uses a more efficient and exact solution.

    Returns:
        A tensor of shape (batch_size, input_sequence_length) representing the
        attention distributions for each sequence in the batch.

    Raises:
        ValueError: mode is not one of ``"recursive"``, ``"parallel"``,
            ``"hard"``.
    """
    # Force things to be tensors
    if not isinstance(p_choose_i, torch.Tensor):
        p_choose_i = torch.tensor(p_choose_i)

    if not isinstance(previous_attention, torch.Tensor):
        previous_attention = torch.tensor(previous_attention)

    if mode == "recursive":
        # Use .shape[0] when it's not None, or fall back on symbolic shape
        batch_size = p_choose_i.shape[0]
        # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ...,
        # 1 - p_choose_i[-2]]
        shifted_1mp_choose_i = torch.cat((p_choose_i.new_ones(batch_size, 1),
                                          1 - p_choose_i[:, :-1]), 1)

        # Compute attention distribution recursively as
        # q[i] = (1 - p_choose_i[i - 1])*q[i - 1] + previous_attention[i]
        # attention[i] = p_choose_i[i]*q[i]

        def f(x, yz):
            return torch.reshape(yz[0] * x + yz[1], (batch_size,))

        x_tmp = f(torch.zeros((batch_size,)), torch.transpose(
            shifted_1mp_choose_i, 0, 1))
        x_tmp = f(x_tmp, torch.transpose(previous_attention, 0, 1))
        attention = p_choose_i * torch.transpose(x_tmp, 0, 1)
    elif mode == "parallel":
        batch_size = p_choose_i.shape[0]
        shifted_1mp_choose_i = torch.cat((p_choose_i.new_ones(batch_size, 1),
                                          1 - p_choose_i[:, :-1]), 1)
        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp_choose_i = safe_cumprod(shifted_1mp_choose_i, dim=1)
        # Compute recurrence relation solution
        attention = p_choose_i * cumprod_1mp_choose_i * torch.cumsum(
            previous_attention / cumprod_1mp_choose_i.clamp(min=1e-10, max=1.),
            dim=1)
    elif mode == "hard":
        # Remove any probabilities before the index chosen last time step
        p_choose_i *= torch.cumsum(previous_attention, dim=1)
        # Now, use exclusive cumprod to remove probabilities after the first
        # chosen index, like so:
        # p_choose_i = [0, 0, 0, 1, 1, 0, 1, 1]
        # cumprod(1 - p_choose_i, exclusive=True) = [1, 1, 1, 1, 0, 0, 0, 0]
        # Product of above: [0, 0, 0, 1, 0, 0, 0, 0]
        batch_size = p_choose_i.shape[0]
        shifted_1mp_choose_i = torch.cat((p_choose_i.new_ones(batch_size, 1),
                                          1 - p_choose_i[:, :-1]), 1)
        attention = p_choose_i * torch.cumprod(shifted_1mp_choose_i, dim=1)
    else:
        raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
    return attention


def _monotonic_probability_fn(score: torch.Tensor,
                              previous_alignments: torch.Tensor,
                              sigmoid_noise: float,
                              mode: str) -> torch.Tensor:
    r"""Attention probability function for monotonic attention.
    Takes in unnormalized attention scores, adds pre-sigmoid noise to
    encourage the model to make discrete attention decisions, passes them
    through a sigmoid to obtain "choosing" probabilities, and then calls
    monotonic_attention to obtain the attention distribution.  For more
    information, see
    `Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  https://arxiv.org/abs/1704.00784`_

    Args:
        score: Unnormalized attention scores, shape
            ``[batch_size, alignments_size]``
        previous_alignments: Previous attention distribution, shape
            ``[batch_size, alignments_size]``
        sigmoid_noise: Standard deviation of pre-sigmoid noise.  Setting this
            larger than 0 will encourage the model to produce large attention
            scores, effectively making the choosing probabilities discrete and
            the resulting attention distribution one-hot.  It should be set to 0
            at test-time, and when hard attention is not desired.
        mode: How to compute the attention distribution.  Must be one of
            ``"recursive"``, ``"parallel"``, or ``"hard"``. Refer to
            :func:`~texar.torch.core.monotonic_attention` for more information.

    Returns:
        A ``[batch_size, alignments_size]`` shaped tensor corresponding to the
        resulting attention distribution.
    """
    # Optionally add pre-sigmoid noise to the scores
    if sigmoid_noise > 0:
        noise = torch.randn(score.shape, dtype=score.dtype, device=score.device)
        score += sigmoid_noise * noise
    # Compute "choosing" probabilities from the attention scores
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = (score > 0).type(score.dtype)
    else:
        p_choose_i = torch.sigmoid(score)
    # Convert from choosing probabilities to attention distribution
    return monotonic_attention(p_choose_i, previous_alignments, mode)


class MonotonicAttentionMechanism(AttentionMechanism, ABC):
    r"""Base attention mechanism for monotonic attention.

    Simply overrides the initial_alignments function to provide a dirac
    distribution, which is needed in order for the monotonic attention
    distributions to have the correct behavior.
    """

    def initial_alignments(self,
                           batch_size: int,
                           max_time: int,
                           dtype: torch.dtype,
                           device: torch.device) -> torch.Tensor:
        r"""Creates the initial alignment values for the monotonic attentions.

        Initializes to dirac distributions, i.e. [1, 0, 0, ...memory length
        ..., 0] for all entries in the batch.

        Args:
            batch_size: integer scalar, the batch_size.
            max_time: integer scalar, the max_time (length of the source
                sequence).
            dtype: The `torch.dtype`.
            device: The `torch.device`.

        Returns:
            A ``dtype`` tensor shaped ``[batch_size, alignments_size]``
            (``alignments_size`` is the value of ``max_time``).
        """
        labels = torch.zeros((batch_size,), dtype=torch.int64,
                             device=device)
        one_hot = torch.eye(max_time, dtype=torch.int64)
        return F.embedding(labels, one_hot)


class BahdanauMonotonicAttention(MonotonicAttentionMechanism):
    r"""Monotonic attention mechanism with Bahdanau-style energy function.
    This type of attention enforces a monotonic constraint on the attention
    distributions; that is once the model attends to a given point in the
    memory it can't attend to any prior points at subsequence output
    time steps.  It achieves this by using the :func:`_monotonic_probability_fn`
    instead of softmax to construct its attention distributions.  Since the
    attention scores are passed through a sigmoid, a learnable scalar bias
    parameter is applied after the score function and before the sigmoid.
    Otherwise, it is equivalent to BahdanauAttention.  This approach is
    proposed in:
    `Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  <https://arxiv.org/abs/1704.00784>`_

    Args:
        num_units: The depth of the query mechanism.
        decoder_output_size: The output size of the decoder cell.
        encoder_output_size: The output size of the encoder cell.
        normalize: Python boolean.  Whether to normalize the energy term.
        score_mask_value: (optional): The mask value for score before
            passing into ``probability_fn``. The default is -inf. Only used
            if :attr:`memory_sequence_length` is not None.
        sigmoid_noise: Standard deviation of pre-sigmoid noise.  Refer to
            :func:`_monotonic_probability_fn` for more
            information.
        score_bias_init: Initial value for score bias scalar.  It's
            recommended to initialize this to a negative value when the
            length of the memory is large.
        mode: How to compute the attention distribution.  Must be one of
            ``"recursive"``, ``"parallel"``, or ``"hard"``.  Refer to
            :func:`~texar.torch.core.monotonic_attention` for more information.
    """

    def __init__(self,
                 num_units: int,
                 decoder_output_size: int,
                 encoder_output_size: int,
                 normalize: bool = False,
                 score_mask_value: Optional[torch.Tensor] = None,
                 sigmoid_noise: float = 0.,
                 score_bias_init: float = 0.,
                 mode: str = "parallel"):
        # Set up the monotonic probability fn with supplied parameters
        self.wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn,
            sigmoid_noise=sigmoid_noise,
            mode=mode)

        super().__init__(
            encoder_output_size=encoder_output_size,
            query_layer=nn.Linear(decoder_output_size, num_units, False),
            memory_layer=nn.Linear(encoder_output_size, num_units, False),
            score_mask_value=score_mask_value)

        limit = np.sqrt(3. / num_units)
        self.attention_v = 2 * limit * torch.rand(num_units) - limit
        self.attention_v = nn.Parameter(self.attention_v,
                                        requires_grad=True)

        self.attention_g: Optional[torch.Tensor]
        self.attention_b: Optional[torch.Tensor]
        if normalize:
            self.attention_g = torch.sqrt(torch.tensor(1. / num_units))
            self.attention_g = nn.Parameter(self.attention_g,
                                            requires_grad=True)
            self.attention_b = torch.zeros(num_units)
            self.attention_b = nn.Parameter(self.attention_b,
                                            requires_grad=True)
        else:
            self.attention_g = None
            self.attention_b = None

        if not isinstance(score_bias_init, torch.Tensor):
            self.attention_score_bias = torch.tensor(score_bias_init)
        self.attention_score_bias = nn.Parameter(self.attention_score_bias)

    def forward(self,  # type: ignore
                query: torch.Tensor,
                state: torch.Tensor,
                memory: torch.Tensor,
                memory_sequence_length: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        query = self._process_query_and_memory(
            query, memory, memory_sequence_length)

        score = _bahdanau_score(query,
                                self._keys,
                                self.attention_v,
                                self.attention_g,
                                self.attention_b)
        score += self.attention_score_bias

        alignments = self.wrapped_probability_fn(
            maybe_mask_score(score, self.score_mask_value,
                             memory_sequence_length), state)

        next_state = alignments
        return alignments, next_state


class LuongMonotonicAttention(MonotonicAttentionMechanism):
    r"""Monotonic attention mechanism with Luong-style energy function.
    This type of attention enforces a monotonic constraint on the attention
    distributions; that is once the model attends to a given point in the
    memory it can't attend to any prior points at subsequence output
    time steps.  It achieves this by using :func:`_monotonic_probability_fn`
    instead of softmax to construct its attention distributions.  Otherwise,
    it is equivalent to LuongAttention.  This approach is proposed in:
    `Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  <https://arxiv.org/abs/1704.00784>`_

    Args:
        num_units: The depth of the query mechanism.
        encoder_output_size: The output size of the encoder cell.
        scale: Python boolean.  Whether to scale the energy term.
        score_mask_value: (optional): The mask value for score before
            passing into ``probability_fn``. The default is -inf. Only used
            if :attr:`memory_sequence_length` is not None.
        sigmoid_noise: Standard deviation of pre-sigmoid noise.  Refer to
            :func:`_monotonic_probability_fn` for more
            information.
        score_bias_init: Initial value for score bias scalar.  It's
            recommended to initialize this to a negative value when the
            length of the memory is large.
        mode: How to compute the attention distribution.  Must be one of
            ``"recursive"``, ``"parallel"``, or ``"hard"``.  Refer to
            :func:`~texar.torch.core.monotonic_attention` for more information.
    """

    def __init__(self,
                 num_units: int,
                 encoder_output_size: int,
                 scale: bool = False,
                 score_mask_value: Optional[torch.Tensor] = None,
                 sigmoid_noise: float = 0.,
                 score_bias_init: float = 0.,
                 mode: str = "parallel"):
        # Set up the monotonic probability fn with supplied parameters
        self.wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn,
            sigmoid_noise=sigmoid_noise,
            mode=mode)

        super().__init__(
            encoder_output_size=encoder_output_size,
            query_layer=None,
            memory_layer=nn.Linear(encoder_output_size, num_units, False),
            score_mask_value=score_mask_value)

        self.attention_g: Optional[torch.Tensor]
        if scale:
            self.attention_g = nn.Parameter(
                torch.tensor(1.0, requires_grad=True))
        else:
            self.attention_g = None

        if not isinstance(score_bias_init, torch.Tensor):
            self.attention_score_bias = torch.tensor(score_bias_init)
        self.attention_score_bias = nn.Parameter(self.attention_score_bias)

    def forward(self,  # type: ignore
                query: torch.Tensor,
                state: torch.Tensor,
                memory: torch.Tensor,
                memory_sequence_length: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        query = self._process_query_and_memory(
            query, memory, memory_sequence_length)

        score = _luong_score(query, self._keys, self.attention_g)
        score += self.attention_score_bias

        alignments = self.wrapped_probability_fn(
            maybe_mask_score(score, self.score_mask_value,
                             memory_sequence_length), state)
        next_state = alignments
        return alignments, next_state


def compute_attention(attention_mechanism: AttentionMechanism,
                      cell_output: torch.Tensor,
                      attention_state: torch.Tensor,
                      memory: torch.Tensor,
                      attention_layer: Optional[nn.Module],
                      memory_sequence_length: Optional[torch.LongTensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Computes the attention and alignments for a given
    :attr:`attention_mechanism`.

    Args:
        attention_mechanism: The :class:`~texar.torch.core.AttentionMechanism`
            instance used to compute attention.
        cell_output (tensor): The decoder output (query tensor), shaped
            ``[batch_size, query_depth]``.
        attention_state (tensor): tensor, shaped
            ``[batch_size, alignments_size]`` (``alignments_size`` is memory's
            ``max_time``).
        memory (tensor): the memory to query; usually the output of an RNN
            encoder. This tensor should be shaped
            ``[batch_size, max_time, ...]``.
        attention_layer (:torch_nn:`Module`, optional): If specified, the
            attention context is concatenated with :attr:`cell_output`, and
            fed through this layer.
        memory_sequence_length (tensor, optional): sequence lengths for the
            batch entries in memory.  If provided, the memory tensor rows are
            masked with zeros for values past the respective sequence lengths.

    Returns:
        A tuple of `(attention, alignments, next_attention_state)`, where

        - ``attention``: The attention context (or the output of
          :attr:`attention_layer`, if specified).
        - ``alignments``: The computed attention alignments.
        - ``next_attention_state``: The attention state after the current time
          step.
    """
    alignments, next_attention_state = attention_mechanism(
        query=cell_output,
        state=attention_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = torch.unsqueeze(alignments, dim=1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = torch.matmul(expanded_alignments, attention_mechanism.values)
    context = torch.squeeze(context, dim=1)

    if attention_layer is not None:
        attention = attention_layer(torch.cat((cell_output, context), dim=1))
    else:
        attention = context

    return attention, alignments, next_attention_state


class AttentionWrapperState(NamedTuple):
    r"""A `namedtuple` storing the state of an
    :class:`~texar.torch.core.AttentionWrapper`.
    """
    cell_state: MaybeList[MaybeTuple[torch.Tensor]]
    r"""The state of the wrapped `RNNCell` at the previous time step."""
    attention: torch.Tensor
    r"""The attention emitted at the previous time step."""
    time: int
    r"""The current time step."""
    alignments: MaybeTuple[torch.Tensor]
    r"""A single or tuple of tensor(s) containing the alignments emitted at
    the previous time step for each attention mechanism."""
    alignment_history: MaybeTuple[List[torch.Tensor]]
    r"""(If enabled) A single or tuple of list(s) containing alignment matrices
    from all time steps for each attention mechanism. Call :torch:`stack` on
    each list to convert to a :tensor:`Tensor`."""
    attention_state: MaybeTuple[torch.Tensor]
    r"""A single or tuple of nested objects containing attention mechanism
    states for each attention mechanism."""
