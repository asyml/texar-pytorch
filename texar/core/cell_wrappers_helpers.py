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

from texar.utils.utils import map_structure, sequence_mask

__all__ = [
    "AttentionMechanism"
]


class AttentionMechanism(object):

    @property
    def alignments_size(self):
        raise NotImplementedError

    @property
    def state_size(self):
        raise NotImplementedError


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
        self.dtype = memory_layer.dtype
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
        self._keys = (
            self.memory_layer(self._values) if self.memory_layer
            else self._values)
        if custom_key_value_fn is not None:
            self._keys, self._valyes = custom_key_value_fn(self._keys,
                                                           self._values)
        self._batch_size = (self._keys.shape[0])
        self._alignments_size = self._keys.shape[0]

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


def _prepare_memory(memory,
                    memory_sequence_length=None,
                    memory_mask=None,
                    check_inner_dims_defined=True):
    """Convert to tensor and possibly mask `memory`.
    Args:
      memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
      memory_mask: `boolean` tensor with shape [batch_size, max_time]. The
      memory should be skipped when the corresponding mask is False.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
      argument's shape is checked to ensure all but the two outermost dimensions
      are fully defined.

    Returns:
      A (possibly masked), checked, new `memory`.

    Raises:
      ValueError: If `check_inner_dims_defined` is `True` and not
        `memory.shape[2:].is_fully_defined()`.
    """
    if memory_sequence_length is not None and memory_mask is not None:
        raise ValueError(
            "memory_sequence_length and memory_mask can't be provided "
            "at same time.")
    if memory_sequence_length is not None:
        memory_sequence_length = torch.as_tensor(memory_sequence_length)
    if check_inner_dims_defined:

        def _check_dim(m):
            for i in m.shape[2:]:
                if i is None:
                    raise ValueError(
                        "Expected memory to have fully defined inner dims, "
                        "but saw shape: {}".format(m.shape))

        _check_dim(memory)
    if memory_sequence_length is not None and memory_mask is None:
        return memory
    elif memory_sequence_length is not None:
        seq_len_mask = sequence_mask(memory_sequence_length,
                                     max_len=memory.shape[1],
                                     dtype=memory.dtype)
    else:
        seq_len_mask = memory_mask.type(memory.dtype)

    def _maybe_mask(m, seq_len_mask):
        """Mask the memory based on the memory mask."""
        rank = m.dim()
        extra_ones = torch.ones(rank-2, dtype=torch.int32)
        seq_len_mask = torch.reshape(
            seq_len_mask,
            torch.cat((seq_len_mask.shape, extra_ones), 0))
        return m * seq_len_mask
    return _maybe_mask(memory, seq_len_mask)


def _maybe_mask_score(score,
                      memory_sequence_length=None,
                      memory_mask=None,
                      score_mask_value=None):
    """Mask the attention score based on the masks."""
    if memory_sequence_length is None and memory_mask is None:
        return score
    if memory_sequence_length is not None and memory_mask is not None:
        raise ValueError(
            "memory_sequence_length and memory_mask can't be provided "
            "at same time.")
    if memory_sequence_length is not None:
        if memory_sequence_length <= 0:
            raise ValueError(
                "All values in memory_sequence_length must be greater "
                "than zero.")
        memory_mask = sequence_mask(memory_sequence_length,
                                    max_len=score.shape[1])
    score_mask_values = score_mask_value * torch.ones_like(score)
    return torch.where(memory_mask, score, score_mask_values)


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
