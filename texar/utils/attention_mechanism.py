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

from typing import NamedTuple, Optional, Tuple, Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from texar.utils.utils import map_structure, sequence_mask

__all__ = [
    "AttentionMechanism",
    "AttentionWrapperState",
    "LuongAttention",
    "BahdanauAttention",
    "hardmax",
    "safe_cumprod",
    "monotonic_attention",
    "BahdanauMonotonicAttention",
    "LuongMonotonicAttention",
]


class AttentionMechanism(object):

    @property
    def alignments_size(self):
        raise NotImplementedError

    @property
    def state_size(self):
        raise NotImplementedError


def _prepare_memory(memory,
                    memory_sequence_length):
    """Convert to tensor and possibly mask `memory`.
    Args:
      memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.

    Returns:
      A (possibly masked), checked, new `memory`.

    Raises:
      ValueError: If `check_inner_dims_defined` is `True` and not
        `memory.shape[2:].is_fully_defined()`.
    """
    if memory_sequence_length is not None:
        memory_sequence_length = torch.tensor(memory_sequence_length)

    if memory_sequence_length is None:
        seq_len_mask = None
    else:
        seq_len_mask = sequence_mask(memory_sequence_length,
                                     max_len=memory.shape[1],
                                     dtype=memory.dtype)
        seq_len_batch_size = memory_sequence_length.shape[0]

    def _maybe_mask(m, seq_len_mask):
        """Mask the memory based on the memory mask."""
        rank = m.dim()
        extra_ones = torch.ones(rank-2, dtype=torch.int32)
        m_batch_size = m.shape[0]

        if memory_sequence_length is not None:
            if seq_len_batch_size != m_batch_size:
                raise ValueError("memory_sequence_length and memory tensor "
                                 "batch sizes do not match.")
            seq_len_mask = torch.reshape(
                seq_len_mask,
                torch.cat((seq_len_mask.shape, extra_ones), 0))
            return m * seq_len_mask
        else:
            return m

    return _maybe_mask(memory, seq_len_mask)


def _maybe_mask_score(score,
                      memory_sequence_length,
                      score_mask_value):
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
                 query_layer,
                 memory,
                 probability_fn,
                 memory_sequence_length=None,
                 memory_layer=None,
                 check_inner_dims_defined=True,
                 score_mask_value=None,
                 custom_key_value_fn=None):
        """Construct base AttentionMechanism class.

        Args:
          query_layer: Callable.  Instance of `tf.compat.v1.layers.Layer`.  The
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
          check_inner_dims_defined: Python boolean.  If `True`, the `memory`
            argument's shape is checked to ensure all but the two outermost
            dimensions are fully defined.
          score_mask_value: (optional): The mask value for score before passing
            into `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          custom_key_value_fn: (optional): The custom function for
            computing keys and values.
        """
        self._query_layer = query_layer
        self._memory_layer = memory_layer
        if not callable(probability_fn):
            raise TypeError("probability_fn must be callable, saw type: %s" %
                            type(probability_fn).__name__)
        if score_mask_value is None:
            score_mask_value = torch.tensor(-np.inf)
        self._probability_fn = lambda score, prev: (
            probability_fn(
                _maybe_mask_score(
                    score,
                    memory_sequence_length=memory_sequence_length,
                    score_mask_value=score_mask_value), prev))
        self._values = _prepare_memory(
            memory,
            memory_sequence_length=memory_sequence_length,
            check_inner_dims_defined=check_inner_dims_defined)
        args = [self._values.shape[-1]] + self.memory_layer.get("config")
        self._keys = (
            self.memory_layer.get("name")(*args)(self._values) if
            self.memory_layer else self._values)
        if custom_key_value_fn is not None:
            self._keys, self._values = custom_key_value_fn(self._keys,
                                                           self._values)
        self._batch_size = (self._keys.shape[0])
        self._alignments_size = self._keys.shape[1]

    @property
    def memory_layer(self):
        return self._memory_layer

    @property
    def query_layer(self):
        return self._query_layer

    @property
    def values(self):
        return self._values

    @property
    def keys(self):
        return self._keys

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self._alignments_size

    def initial_alignments(self, batch_size, dtype):
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
        return torch.zeros(batch_size, max_time, dtype=dtype)

    def initial_state(self, batch_size, dtype):
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
        return self.initial_alignments(batch_size, dtype)


def _luong_score(query, keys, scale):
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
            "Incompatible or unknown inner dimensions between query and keys.  "
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
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 custom_key_value_fn=None):
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
          dtype: The data type for the memory layer of the attention mechanism.
          custom_key_value_fn: (optional): The custom function for
            computing keys and values.
        """
        # For LuongAttention, we only transform the memory layer; thus
        # num_units **must** match expected the query depth.
        if probability_fn is None:
            probability_fn = F.softmax
        if dtype is None:
            dtype = torch.float32
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(LuongAttention, self).__init__(
            query_layer=None,
            memory_layer={"name": nn.Linear,
                          "config": [num_units, False]},
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            custom_key_value_fn=custom_key_value_fn)
        self._num_units = num_units
        self._scale = scale

    def __call__(self, query, state):
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
        attention_g = None
        if self._scale:
            attention_g = torch.tensor(1)
        score = _luong_score(query, self._keys, attention_g)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


def _bahdanau_score(processed_query,
                    keys,
                    attention_v,
                    attention_g=None,
                    attention_b=None):
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
      processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      attention_v: Tensor, shape `[num_units]`.
      attention_g: Optional scalar tensor for normalization.
      attention_b: Optional tensor with shape `[num_units]` for normalization.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = torch.unsqueeze(processed_query, 1)
    if attention_g is not None and attention_b is not None:
        normed_v = attention_g * attention_v * \
                   torch.rsqrt(torch.sum(attention_v**2))
        return torch.sum(normed_v * F.tanh(keys + processed_query
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
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 custom_key_value_fn=None):
        """Construct the Attention mechanism.

            Args:
              num_units: The depth of the query mechanism.
              memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
              memory_sequence_length: (optional) Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
              normalize: Python boolean.  Whether to normalize the energy term.
              probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is `tf.nn.softmax`. Other options include
                `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
                Its signature should be: `probabilities = probability_fn(score)`.
              score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
              dtype: The data type for the query and memory layers of the attention
                mechanism.
              custom_key_value_fn: (optional): The custom function for
                computing keys and values.
        """
        if probability_fn is None:
            probability_fn = F.softmax
        if dtype is None:
            dtype = torch.float32
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(BahdanauAttention, self).__init__(
            query_layer={"name": nn.Linear,
                         "config": [num_units, False]},
            memory_layer={"name": nn.Linear,
                          "config": [num_units, False]},
            memory=memory,
            probability_fn=wrapped_probability_fn,
            custom_key_value_fn=custom_key_value_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value)
        self._num_units = num_units
        self._normalize = normalize

    def __call__(self, query, state):
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
        args = [query.shape[-1]] + self.query_layer.get("config")
        processed_query = self.query_layer.get("name")(*args)(query) \
            if self.query_layer else query
        attention_v =






















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
    alignments: torch.Tensor
    alignment_history: Any
    attention_state: Any

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
            return new

        return with_same_shape(
            self,
            super(AttentionWrapperState, self)._replace(**kwargs))








def hardmax(logits):
    """Returns batched one-hot vectors.

    The depth index containing the `1` is that of the maximum logit value.

    Args:
      logits: A batch tensor of logit values.

    Returns:
      A batched one-hot tensor.
    """
    depth = logits.shape[-1]
    return F.one_hot(torch.argmax(logits, -1), depth)


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    if isinstance(attention_mechanism, _BaseAttentionMechanismV2):
        alignments, next_attention_state = attention_mechanism(
            [cell_output, attention_state])
    else:
        # For other class, assume they are following _BaseAttentionMechanism,
        # which takes query and state as separate parameter.
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
    context_ = torch.matmul(expanded_alignments, attention_mechanism.values)
    context_ = torch.squeeze(context_, dim=1)

    if attention_layer is not None:
        attention_input = torch.cat((cell_output, context_), dim=1)
        _attention_layer = attention_layer.get("layer_name")
        in_features = attention_input.shape[-1]
        attention = _attention_layer(in_features,
                                     attention_layer.get("out_features"),
                                     attention_layer.get("bias"))
        attention = attention(attention_input)
    else:
        attention = context_

    return attention, alignments, next_attention_state
