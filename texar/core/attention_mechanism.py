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
Various helper classes and utilities for cell wrappers.
"""

# pylint: disable=too-many-arguments, too-many-instance-attributes
# pylint: disable=missing-docstring  # does not support generic classes

from typing import NamedTuple, Optional, List, Union, Callable, Tuple, TypeVar

import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from texar.module_base import ModuleBase
from texar.utils.utils import sequence_mask
from texar.utils.types import MaybeTuple

__all__ = [
    "AttentionMechanism",
    "AttentionWrapperState",
    "LuongAttention",
    "BahdanauAttention",
    "hardmax",
    "safe_cumprod",
    "compute_attention",
    "monotonic_attention",
    "BahdanauMonotonicAttention",
    "LuongMonotonicAttention",
]

State = TypeVar('State')


class AttentionMechanism(ModuleBase):

    @property
    def memory_layer(self):
        raise NotImplementedError

    @property
    def query_layer(self):
        raise NotImplementedError

    @property
    def values(self):
        raise NotImplementedError

    @property
    def keys(self):
        raise NotImplementedError

    @property
    def batch_size(self):
        raise NotImplementedError

    @property
    def alignments_size(self):
        raise NotImplementedError

    @property
    def state_size(self):
        raise NotImplementedError


def _prepare_memory(memory: torch.Tensor,
                    memory_sequence_length: Optional[torch.LongTensor]) -> \
        torch.Tensor:
    """Convert to tensor and possibly mask `memory`.

    Args:
      memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.

    Returns:
      A (possibly masked), new `memory`.
    """
    if memory_sequence_length is not None and \
            not isinstance(memory_sequence_length, torch.Tensor):
        memory_sequence_length = torch.tensor(memory_sequence_length)

    if memory_sequence_length is None:
        seq_len_mask = None
    else:
        seq_len_mask = sequence_mask(memory_sequence_length,
                                     max_len=memory.shape[1],
                                     dtype=memory.dtype)
        seq_len_batch_size = memory_sequence_length.shape[0]

    """Mask the memory based on the memory mask."""
    rank = memory.dim()
    extra_ones = [1] * (rank - 2)
    m_batch_size = memory.shape[0]

    if seq_len_mask is not None:
        if seq_len_batch_size != m_batch_size:
            raise ValueError("memory_sequence_length and memory tensor "
                             "batch sizes do not match.")
        seq_len_mask = torch.reshape(
            seq_len_mask,
            tuple(list(seq_len_mask.shape) + extra_ones))
        return memory * seq_len_mask
    else:
        return memory


def _maybe_mask_score(score: torch.Tensor,
                      memory_sequence_length: Optional[torch.LongTensor],
                      score_mask_value: torch.Tensor) -> torch.Tensor:
    """Mask the attention score based on the masks."""
    if memory_sequence_length is None:
        return score

    for memory_sequence_length_value in memory_sequence_length:
        if memory_sequence_length_value <= 0:
            raise ValueError(
                "All values in memory_sequence_length must be greater "
                "than zero.")

    score_mask = sequence_mask(memory_sequence_length,
                               max_len=score.shape[1])
    score_mask_values = score_mask_value * torch.ones_like(score)
    return torch.where(score_mask, score, score_mask_values)


class _BaseAttentionMechanism(AttentionMechanism):
    """A base AttentionMechanism class providing common functionality.

    Common functionality includes:
      1. Storing the query and memory layers.
      2. Preprocessing and storing the memory.
    """
    def __init__(self,
                 query_layer: Optional[nn.Module],
                 memory: torch.Tensor,
                 probability_fn: Callable[[torch.Tensor,
                                           torch.Tensor], torch.Tensor],
                 memory_sequence_length: Optional[torch.LongTensor] = None,
                 memory_layer: Optional[nn.Module] = None,
                 score_mask_value: Optional[torch.Tensor] = None):
        """Construct base AttentionMechanism class.

        Args:
          query_layer: Callable. Instance of `tf.compat.v1.layers.Layer`. The
            layer's depth must match the depth of `memory_layer`.  If
            `query_layer` is not provided, the shape of `query` must match that
            of `memory_layer`.
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          probability_fn: A `callable`.  Converts the score and previous
            alignments to probabilities. Its signature should be:
            `probabilities = probability_fn(score, state)`.
          memory_sequence_length (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          memory_layer: Instance of `tf.compat.v1.layers.Layer` (may be None).
            The layer's depth must match the depth of `query_layer`. If
            `memory_layer` is not provided, the shape of `memory` must match
            that of `query_layer`.
          score_mask_value: (optional): The mask value for score before passing
            into `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
        """
        super().__init__()

        if (query_layer is not None and
                not isinstance(query_layer, torch.nn.modules.linear.Linear)):
            raise TypeError("query_layer is not a Linear Layer: %s"
                            % type(query_layer).__name__)
        if (memory_layer is not None and
                not isinstance(memory_layer, torch.nn.modules.linear.Linear)):
            raise TypeError("memory_layer is not a Linear Layer: %s"
                            % type(memory_layer).__name__)
        self._query_layer = query_layer
        self._memory_layer = memory_layer

        if not callable(probability_fn):
            raise TypeError("probability_fn must be callable, saw type: %s" %
                            type(probability_fn).__name__)
        if score_mask_value is None:
            score_mask_value = torch.tensor(-np.inf)
        self._probability_fn = lambda score, prev: (
            probability_fn(
                _maybe_mask_score(score, memory_sequence_length,
                                  score_mask_value), prev))
        self._values = _prepare_memory(memory, memory_sequence_length)

        self._keys: torch.Tensor

        if self.memory_layer is not None:
            self._keys = self.memory_layer(self._values)
        else:
            self._keys = self._values

        self._batch_size = self._keys.shape[0]
        self._alignments_size = self._keys.shape[1]

    @property
    def memory_layer(self) -> Optional[nn.Module]:
        return self._memory_layer

    @property
    def query_layer(self) -> Optional[nn.Module]:
        return self._query_layer

    @property
    def values(self) -> torch.Tensor:
        return self._values

    @property
    def keys(self) -> torch.Tensor:
        return self._keys

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def alignments_size(self) -> int:
        return self._alignments_size

    @property
    def state_size(self) -> int:
        return self._alignments_size

    def initial_alignments(self,
                           batch_size: int) -> torch.Tensor:
        """Creates the initial alignment values for the `AttentionWrapper`
        class.

        This is important for AttentionMechanisms that use the previous
        alignment to calculate the alignment at the next time step
        (e.g. monotonic attention).

        The default behavior is to return a tensor of all zeros.

        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.

        Returns:
          A `dtype` tensor shaped `[batch_size, alignments_size]`
          (`alignments_size` is the values' `max_time`).
        """
        max_time = self._alignments_size
        return self._keys.new_zeros((batch_size, max_time))

    def initial_state(self,
                      batch_size: int) -> torch.Tensor:
        """Creates the initial state values for the `AttentionWrapper` class.

        This is important for AttentionMechanisms that use the previous
        alignment to calculate the alignment at the next time step
        (e.g. monotonic attention).

        The default behavior is to return the same output as initial_alignments.

        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.

        Returns:
          A structure of all-zero tensors with shapes as described by
          `state_size`.
        """
        return self.initial_alignments(batch_size)


def _luong_score(query: torch.Tensor,
                 keys: torch.Tensor,
                 scale: Optional[torch.Tensor]) -> torch.Tensor:
    """Implements Luong-style (multiplicative) scoring function.

    This attention has two forms.  The first is standard Luong attention,
    as described in:

    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.  https://arxiv.org/abs/1508.04025

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.

    To enable the second form, call this function with `scale=True`.

    Args:
      query: Tensor, shape `[batch_size, num_units]` to compare to keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      scale: the optional tensor to scale the attention score.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.

    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = query.shape[-1]
    key_units = keys.shape[-1]
    if depth != key_units:
        raise ValueError(
            "Incompatible or unknown inner dimensions between query and keys. "
            "Query (%s) has units: %s.  Keys (%s) have units: %s. "
            "Perhaps you need to set num_units to the keys' dimension (%s)?" %
            (query, depth, keys, key_units, key_units))

    # Reshape from [batch_size, depth] to [batch_size, 1, depth]
    # for matmul.
    query = torch.unsqueeze(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                keys is [batch_size, max_time, depth].
    # the inner product is asked to **transpose keys' inner shape** to get a
    # batched matmul on:
    #   [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of:
    #   [batch_size, 1, max_time].
    # we then squeeze out the center singleton dimension.
    score = torch.matmul(query, keys.permute(0, 2, 1))
    score = torch.squeeze(score, 1)

    if scale is not None:
        # Scalar used in weight scaling
        score = scale * score
    return score


class LuongAttention(_BaseAttentionMechanism):
    """Implements Luong-style (multiplicative) attention scoring.

    This attention has two forms.  The first is standard Luong attention,
    as described in:

    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    [Effective Approaches to Attention-based Neural Machine Translation.
    EMNLP 2015.](https://arxiv.org/abs/1508.04025)

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.

    To enable the second form, construct the object with parameter
    `scale=True`.
    """

    def __init__(self,
                 num_units: int,
                 memory: torch.Tensor,
                 memory_sequence_length: Optional[torch.LongTensor] = None,
                 scale: bool = False,
                 probability_fn: Optional[Callable[[torch.Tensor],
                                                   torch.Tensor]] = None,
                 score_mask_value: Optional[torch.Tensor] = None):
        """Construct the AttentionMechanism mechanism.

        Args:
          num_units: The depth of the attention mechanism.
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length: (optional) Sequence lengths for the batch
            entries in memory.  If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          scale: Python boolean.  Whether to scale the energy term.
          probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is `tf.nn.softmax`. Other options
            include `tf.contrib.seq2seq.hardmax` and
            `tf.contrib.sparsemax.sparsemax`. Its signature should be:
            `probabilities = probability_fn(score)`.
          score_mask_value: (optional) The mask value for score before passing
            into `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
        """
        # For LuongAttention, we only transform the memory layer; thus
        # num_units **must** match expected the query depth.
        if probability_fn is None:
            probability_fn = lambda x: F.softmax(x, dim=-1)
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(LuongAttention, self).__init__(
            query_layer=None,
            memory_layer=nn.Linear(memory.shape[-1], num_units, False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value)
        self._num_units = num_units
        self._scale = scale

        self.attention_g: Optional[torch.Tensor]
        if self._scale:
            self.attention_g = nn.Parameter(torch.tensor(1.0),
                                            requires_grad=True)
        else:
            self.attention_g = None

    def forward(self,  # type: ignore
                query: torch.Tensor,
                state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        score = _luong_score(query, self._keys, self.attention_g)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


def _bahdanau_score(processed_query: torch.Tensor,
                    keys: torch.Tensor,
                    attention_v: torch.Tensor,
                    attention_g: Optional[torch.Tensor] = None,
                    attention_b: Optional[torch.Tensor] = None):
    """Implements Bahdanau-style (additive) scoring function.

    This attention has two forms.  The first is Bhandanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, set please pass in attention_g and attention_b.
    Args:
      processed_query: Tensor, shape `[batch_size, num_units]` to compare to
      keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      attention_v: Tensor, shape `[num_units]`.
      attention_g: Optional scalar tensor for normalization.
      attention_b: Optional tensor with shape `[num_units]` for normalization.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    processed_query = torch.unsqueeze(processed_query, 1)
    if attention_g is not None and attention_b is not None:
        normed_v = attention_g * attention_v * torch.rsqrt(
            torch.sum(attention_v ** 2))
        return torch.sum(normed_v * torch.tanh(keys + processed_query
                                               + attention_b), 2)
    else:
        return torch.sum(attention_v * torch.tanh(keys + processed_query), 2)


class BahdanauAttention(_BaseAttentionMechanism):
    """Implements Bahdanau-style (additive) attention.

      This attention has two forms.  The first is Bahdanau attention,
      as described in:

      Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
      "Neural Machine Translation by Jointly Learning to Align and Translate."
      ICLR 2015. https://arxiv.org/abs/1409.0473

      The second is the normalized form.  This form is inspired by the
      weight normalization article:

      Tim Salimans, Diederik P. Kingma.
      "Weight Normalization: A Simple Reparameterization to Accelerate
       Training of Deep Neural Networks."
      https://arxiv.org/abs/1602.07868

      To enable the second form, construct the object with parameter
      `normalize=True`.
    """

    def __init__(self,
                 num_units: int,
                 cell_output_size: int,
                 memory: torch.Tensor,
                 memory_sequence_length: Optional[torch.Tensor] = None,
                 normalize: bool = False,
                 probability_fn: Optional[Callable[[torch.Tensor],
                                                   torch.Tensor]] = None,
                 score_mask_value: Optional[torch.Tensor] = None):
        """Construct the Attention mechanism.

            Args:
              num_units: The depth of the query mechanism.
              cell_output_size: The output size of the decoder cell.
              memory: The memory to query; usually the output of an RNN encoder.
                This tensor should be shaped `[batch_size, max_time, ...]`.
              memory_sequence_length: (optional) Sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.
              normalize: Python boolean.  Whether to normalize the energy term.
              probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is `tf.nn.softmax`. Other options
                include `tf.contrib.seq2seq.hardmax` and
                `tf.contrib.sparsemax.sparsemax`. Its signature should be:
                `probabilities = probability_fn(score)`.
              score_mask_value: (optional): The mask value for score before
                passing into `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
        """
        if probability_fn is None:
            probability_fn = lambda x: F.softmax(x, dim=-1)
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(BahdanauAttention, self).__init__(
            query_layer=nn.Linear(cell_output_size, num_units, False),
            memory_layer=nn.Linear(memory.shape[-1], num_units, False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value)
        self._num_units = num_units
        self._normalize = normalize

        num_units = self._keys.shape[2]
        limit = np.sqrt(3. / num_units)
        self.attention_v = 2 * limit * torch.rand(
            num_units, dtype=self.values.dtype) - limit
        self.attention_v = nn.Parameter(self.attention_v,
                                        requires_grad=True)

        self.attention_g: Optional[torch.Tensor]
        self.attention_b: Optional[torch.Tensor]
        if self._normalize:
            self.attention_g = torch.sqrt(torch.tensor(1. / num_units))
            self.attention_g = nn.Parameter(self.attention_g,
                                            requires_grad=True)
            self.attention_b = torch.zeros(num_units,
                                           dtype=self.values.dtype)
            self.attention_b = nn.Parameter(self.attention_b,
                                            requires_grad=True)
        else:
            self.attention_g = None
            self.attention_b = None

    def forward(self,  # type: ignore
                query: torch.Tensor,
                state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score the query based on the keys and values.

            Args:
              query: Tensor of dtype matching `self.values` and shape
              `[batch_size, query_depth]`.
              state: Tensor of dtype matching `self.values` and shape
              `[batch_size, alignments_size]` (`alignments_size` is memory's
              `max_time`).

            Returns:
              alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        processed_query = self.query_layer(query) if self.query_layer else query

        score = _bahdanau_score(processed_query,
                                self._keys,
                                self.attention_v,
                                self.attention_g,
                                self.attention_b)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


def safe_cumprod(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Computes cumprod of x in logspace using cumsum to avoid underflow.

      The cumprod function and its gradient can result in numerical
      instabilities when its argument has very small and/or zero values.
      As long as the argument is all positive, we can instead compute the
      cumulative product as exp(cumsum(log(x))).  This function can be called
      identically to tf.cumprod.

      Args:
        x: Tensor to take the cumulative product of.
        *args: Passed on to cumsum; these are identical to those in cumprod.
        **kwargs: Passed on to cumsum; these are identical to those in cumprod.

      Returns:
        Cumulative product of x.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    type_map = {torch.float16: np.float16,
                torch.float32: np.float32,
                torch.float64: np.float64}

    tiny = np.finfo(type_map[x.dtype]).tiny
    return torch.exp(torch.cumsum(torch.log(torch.clamp(x, tiny, 1)),
                                  *args, **kwargs))


def monotonic_attention(p_choose_i: torch.Tensor,
                        previous_attention: torch.Tensor,
                        mode: str) -> torch.Tensor:
    """Compute monotonic attention distribution from choosing probabilities.

      Monotonic attention implies that the input sequence is processed in an
      explicitly left-to-right manner when generating the output sequence.  In
      addition, once an input sequence element is attended to at a given output
      timestep, elements occurring before it cannot be attended to at subsequent
      output timesteps.  This function generates attention distributions
      according to these assumptions.  For more information, see `Online and
      Linear-Time Attention by Enforcing Monotonic Alignments`.

      Args:
        p_choose_i: Probability of choosing input sequence/memory element i.
          Should be of shape (batch_size, input_sequence_length), and should
          all be in the range [0, 1].
        previous_attention: The attention distribution from the previous output
          timestep.  Should be of shape (batch_size, input_sequence_length).
          For the first output timestep, preevious_attention[n] should be
          [1, 0, 0, ..., 0] for all n in [0, ... batch_size - 1].
        mode: How to compute the attention distribution.  Must be one of
          'recursive', 'parallel', or 'hard'.
            * 'recursive' uses tf.scan to recursively compute the distribution.
              This is slowest but is exact, general, and does not suffer from
              numerical instabilities.
            * 'parallel' uses parallelized cumulative-sum and cumulative-product
              operations to compute a closed-form solution to the recurrence
              relation defining the attention distribution.  This makes it more
              efficient than 'recursive', but it requires numerical checks
              which make the distribution non-exact.  This can be a problem in
              particular when input_sequence_length is long and/or p_choose_i
              has entries very close to 0 or 1.
            * 'hard' requires that the probabilities in p_choose_i are all
              either 0 or 1, and subsequently uses a more efficient and exact
              solution.

      Returns:
        A tensor of shape (batch_size, input_sequence_length) representing the
        attention distributions for each sequence in the batch.

      Raises:
        ValueError: mode is not one of 'recursive', 'parallel', 'hard'.
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
            return torch.reshape(yz[0]*x + yz[1], (batch_size,))

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
        attention = p_choose_i*torch.cumprod(shifted_1mp_choose_i, dim=1)
    else:
        raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
    return attention


def _monotonic_probability_fn(score: torch.Tensor,
                              previous_alignments: torch.Tensor,
                              sigmoid_noise: float,
                              mode: str) -> torch.Tensor:
    """Attention probability function for monotonic attention.

      Takes in unnormalized attention scores, adds pre-sigmoid noise to
      encourage the model to make discrete attention decisions, passes them
      through a sigmoid to obtain "choosing" probabilities, and then calls
      monotonic_attention to obtain the attention distribution.  For more
      information, see

      Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
      "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
      ICML 2017.  https://arxiv.org/abs/1704.00784

      Args:
        score: Unnormalized attention scores, shape
          `[batch_size, alignments_size]`
        previous_alignments: Previous attention distribution, shape
          `[batch_size, alignments_size]`
        sigmoid_noise: Standard deviation of pre-sigmoid noise.  Setting this
          larger than 0 will encourage the model to produce large attention
          scores, effectively making the choosing probabilities discrete and
          the resulting attention distribution one-hot.  It should be set to 0
          at test-time, and when hard attention is not desired.
        mode: How to compute the attention distribution.  Must be one of
          'recursive', 'parallel', or 'hard'.  See the docstring for
          `tf.contrib.seq2seq.monotonic_attention` for more information.

      Returns:
        A `[batch_size, alignments_size]`-shape tensor corresponding to the
        resulting attention distribution.
    """
    # Optionally add pre-sigmoid noise to the scores
    if sigmoid_noise > 0:
        noise = torch.randn(score.shape, dtype=score.dtype, device=score.device)
        score += sigmoid_noise*noise
    # Compute "choosing" probabilities from the attention scores
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = (score > 0).type(score.dtype)
    else:
        p_choose_i = torch.sigmoid(score)
    # Convert from choosing probabilities to attention distribution
    return monotonic_attention(p_choose_i, previous_alignments, mode)


class _BaseMonotonicAttentionMechanism(_BaseAttentionMechanism):
    """Base attention mechanism for monotonic attention.

      Simply overrides the initial_alignments function to provide a dirac
      distribution, which is needed in order for the monotonic attention
      distributions to have the correct behavior.
    """

    def initial_alignments(self,
                           batch_size: int) -> torch.Tensor:
        """Creates the initial alignment values for the monotonic attentions.

            Initializes to dirac distributions, i.e. [1, 0, 0, ...memory length
            ..., 0] for all entries in the batch.

            Args:
              batch_size: `int32` scalar, the batch_size.

            Returns:
              A `dtype` tensor shaped `[batch_size, alignments_size]`
              (`alignments_size` is the values' `max_time`).
        """
        max_time = self._alignments_size
        labels = torch.zeros((batch_size,), dtype=torch.int64,
                             device=self._values.device)
        one_hot = torch.eye(max_time, dtype=torch.int64)
        return F.embedding(labels, one_hot)


class BahdanauMonotonicAttention(_BaseMonotonicAttentionMechanism):
    """Monotonic attention mechanism with Bahadanau-style energy function.

      This type of attention enforces a monotonic constraint on the attention
      distributions; that is once the model attends to a given point in the
      memory it can't attend to any prior points at subsequence output
      timesteps.  It achieves this by using the _monotonic_probability_fn
      instead of softmax to construct its attention distributions.  Since the
      attention scores are passed through a sigmoid, a learnable scalar bias
      parameter is applied after the score function and before the sigmoid.
      Otherwise, it is equivalent to BahdanauAttention.  This approach is
      proposed in

      Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
      "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
      ICML 2017.  https://arxiv.org/abs/1704.00784
    """

    def __init__(self,
                 num_units: int,
                 cell_output_size: int,
                 memory: torch.Tensor,
                 memory_sequence_length: Optional[torch.Tensor] = None,
                 normalize: bool = False,
                 score_mask_value: Optional[torch.Tensor] = None,
                 sigmoid_noise: float = 0.,
                 score_bias_init: float = 0.,
                 mode: str = "parallel"):
        """Construct the Attention mechanism.

            Args:
              num_units: The depth of the query mechanism.
              cell_output_size: The output size of the decoder cell.
              memory: The memory to query; usually the output of an RNN encoder.
                This tensor should be shaped `[batch_size, max_time, ...]`.
              memory_sequence_length (optional): Sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.
              normalize: Python boolean.  Whether to normalize the energy term.
              score_mask_value: (optional): The mask value for score before
                passing into `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
              sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the
                docstring for `_monotonic_probability_fn` for more information.
              score_bias_init: Initial value for score bias scalar.  It's
                recommended to initialize this to a negative value when the
                length of the memory is large.
              mode: How to compute the attention distribution.  Must be one of
                'recursive', 'parallel', or 'hard'.  See the docstring for
                `tf.contrib.seq2seq.monotonic_attention` for more information.
        """
        # Set up the monotonic probability fn with supplied parameters
        wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode)
        super(BahdanauMonotonicAttention, self).__init__(
            query_layer=nn.Linear(cell_output_size, num_units, False),
            memory_layer=nn.Linear(memory.shape[-1], num_units, False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value)
        self._num_units = num_units
        self._normalize = normalize

        num_units = self._keys.shape[2]
        limit = np.sqrt(3. / num_units)
        self.attention_v = 2 * limit * torch.rand(
            num_units,
            dtype=self.values.dtype) - limit
        self.attention_v = nn.Parameter(self.attention_v,
                                        requires_grad=True)

        self.attention_g: Optional[torch.Tensor]
        self.attention_b: Optional[torch.Tensor]
        if self._normalize:
            self.attention_g = torch.sqrt(torch.tensor(1. / num_units))
            self.attention_g = nn.Parameter(self.attention_g,
                                            requires_grad=True)
            self.attention_b = torch.zeros(num_units,
                                           dtype=self.values.dtype)
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
                state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score the query based on the keys and values.

            Args:
              query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
              state: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

            Returns:
              alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        processed_query = self.query_layer(query) if self.query_layer else query

        score = _bahdanau_score(processed_query,
                                self._keys,
                                self.attention_v,
                                self.attention_g,
                                self.attention_b)

        score += self.attention_score_bias
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


class LuongMonotonicAttention(_BaseMonotonicAttentionMechanism):
    """Monotonic attention mechanism with Luong-style energy function.

      This type of attention enforces a monotonic constraint on the attention
      distributions; that is once the model attends to a given point in the
      memory it can't attend to any prior points at subsequence output
      timesteps.  It achieves this by using the _monotonic_probability_fn
      instead of softmax to construct its attention distributions.  Otherwise,
      it is equivalent to LuongAttention.  This approach is proposed in

      Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
      "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
      ICML 2017.  https://arxiv.org/abs/1704.00784
    """

    def __init__(self,
                 num_units: int,
                 memory: torch.Tensor,
                 memory_sequence_length: Optional[torch.Tensor] = None,
                 scale: bool = False,
                 score_mask_value: Optional[torch.Tensor] = None,
                 sigmoid_noise: float = 0.,
                 score_bias_init: float = 0.,
                 mode: str = "parallel"):
        """Construct the Attention mechanism.

            Args:
              num_units: The depth of the query mechanism.
              memory: The memory to query; usually the output of an RNN encoder.
                This tensor should be shaped `[batch_size, max_time, ...]`.
              memory_sequence_length (optional): Sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.
              scale: Python boolean.  Whether to scale the energy term.
              score_mask_value: (optional): The mask value for score before
                passing into `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
              sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the
                docstring for `_monotonic_probability_fn` for more information.
              score_bias_init: Initial value for score bias scalar.  It's
                recommended to initialize this to a negative value when the
                length of the memory is large.
              mode: How to compute the attention distribution.  Must be one of
                'recursive', 'parallel', or 'hard'.  See the docstring for
                `tf.contrib.seq2seq.monotonic_attention` for more information.
        """
        # Set up the monotonic probability fn with supplied parameters
        wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode)
        super(LuongMonotonicAttention, self).__init__(
            query_layer=None,
            memory_layer=nn.Linear(memory.shape[-1], num_units, False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value)
        self._num_units = num_units
        self._scale = scale

        self.attention_g: Optional[torch.Tensor]
        if self._scale:
            self.attention_g = nn.Parameter(
                torch.tensor(1.0, requires_grad=True))
        else:
            self.attention_g = None

        if not isinstance(score_bias_init, torch.Tensor):
            self.attention_score_bias = torch.tensor(score_bias_init)
        self.attention_score_bias = nn.Parameter(self.attention_score_bias)

    def forward(self,  # type: ignore
                query: torch.Tensor,
                state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score the query based on the keys and values.

            Args:
              query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
              state: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

            Returns:
              alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        score = _luong_score(query, self._keys, self.attention_g)
        score += self.attention_score_bias
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


def hardmax(logits: torch.Tensor) -> torch.Tensor:
    """Returns batched one-hot vectors.

    The depth index containing the `1` is that of the maximum logit value.

    Args:
      logits: A batch tensor of logit values.

    Returns:
      A batched one-hot tensor.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    depth = logits.shape[-1]
    one_hot = torch.eye(depth, dtype=torch.int64)
    return F.embedding(torch.argmax(logits, -1), one_hot)


def compute_attention(attention_mechanism: AttentionMechanism,
                      cell_output: torch.Tensor,
                      attention_state: torch.Tensor,
                      attention_layer: Optional[nn.Module]) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

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
    """`namedtuple` storing the state of a `AttentionWrapper`.

    Contains:

      - `cell_state`: The state of the wrapped `RNNCell` at the previous time
        step.
      - `attention`: The attention emitted at the previous time step.
      - `time`: int32 scalar containing the current time step.
      - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
         emitted at the previous time step for each attention mechanism.
      - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
         containing alignment matrices from all time steps for each attention
         mechanism. Call `stack()` on each to convert to a `Tensor`.
      - `attention_state`: A single or tuple of nested objects
         containing attention mechanism state for each attention mechanism.
         The objects may contain Tensors or TensorArrays.
    """
    cell_state: torch.Tensor
    attention: torch.Tensor
    time: int
    alignments: MaybeTuple[torch.Tensor]
    alignment_history: Optional[MaybeTuple[List[torch.Tensor]]]
    attention_state: MaybeTuple[torch.Tensor]

    def clone(self, **kwargs):
        """Clone this object, overriding components provided by kwargs.

        The new state fields' shape must match original state fields' shape.
        This will be validated, and original fields' shape will be propagated
        to new fields.

        Example:

        ```python
        initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
        initial_state = initial_state.clone(cell_state=encoder_state)
        ```

        Args:
          **kwargs: Any properties of the state object to replace in the
          returned `AttentionWrapperState`.

        Returns:
          A new `AttentionWrapperState` whose properties are the same as
          this one, except any overridden properties as provided in `kwargs`.
        """

        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor):
                if old.shape != new.shape:
                    raise ValueError(
                        "The shape of the AttentionWrapperState is "
                        "expected to be same as the one to clone. "
                        "self.shape: {}, input.shape: {}".fotmat
                        (old.shape, new.shape))
            return new

        return with_same_shape(
            self, super(AttentionWrapperState, self)._replace(**kwargs))
