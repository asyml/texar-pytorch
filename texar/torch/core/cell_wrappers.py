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
TensorFlow-style RNN cell wrappers.
"""

from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import nn

from texar.torch.core.attention_mechanism import (
    AttentionMechanism, AttentionWrapperState, compute_attention)
from texar.torch.utils import utils
from texar.torch.utils.types import MaybeList

__all__ = [
    'RNNState',
    'LSTMState',
    'HiddenState',
    'wrap_builtin_cell',
    'RNNCellBase',
    'RNNCell',
    'GRUCell',
    'LSTMCell',
    'DropoutWrapper',
    'ResidualWrapper',
    'HighwayWrapper',
    'MultiRNNCell',
    'AttentionWrapper',
]

State = TypeVar('State')
RNNState = torch.Tensor
LSTMState = Tuple[torch.Tensor, torch.Tensor]

HiddenState = MaybeList[Union[RNNState, LSTMState]]


def wrap_builtin_cell(cell: nn.RNNCellBase):
    r"""Convert a built-in :torch_nn:`RNNCellBase` derived RNN cell to
    our wrapped version.

    Args:
        cell: the RNN cell to wrap around.

    Returns:
        The wrapped cell derived from
        :class:`texar.torch.core.cell_wrappers.RNNCellBase`.
    """
    # convert cls to corresponding derived wrapper class
    if isinstance(cell, nn.RNNCell):
        self = RNNCellBase.__new__(RNNCell)
    elif isinstance(cell, nn.GRUCell):
        self = RNNCellBase.__new__(GRUCell)
    elif isinstance(cell, nn.LSTMCell):
        self = RNNCellBase.__new__(LSTMCell)
    else:
        raise TypeError(f"Unrecognized class {type(cell)}.")
    RNNCellBase.__init__(self, cell)
    return self


class RNNCellBase(nn.Module, Generic[State]):
    r"""The base class for RNN cells in our framework. Major differences over
    :torch_nn:`RNNCell` are two-fold:

    1. Holds an :torch_nn:`Module` which could either be a built-in
       RNN cell or a wrapped cell instance. This design allows
       :class:`RNNCellBase` to serve as the base class for both vanilla
       cells and wrapped cells.

    2. Adds :meth:`zero_state` method for initialization of hidden states,
       which can also be used to implement batch-specific initialization
       routines.
    """

    def __init__(self, cell: Union[nn.RNNCellBase, 'RNNCellBase']):
        super().__init__()
        if not isinstance(cell, nn.Module):
            raise ValueError("Type of parameter 'cell' must be derived from"
                             "nn.Module, and has 'input_size' and 'hidden_size'"
                             "attributes.")
        self._cell = cell

    @property
    def input_size(self) -> int:
        r"""The number of expected features in the input."""
        return self._cell.input_size

    @property
    def hidden_size(self) -> int:
        r"""The number of features in the hidden state."""
        return self._cell.hidden_size

    @property
    def _param(self) -> nn.Parameter:
        r"""Convenience method to access a parameter under the module. Useful
        when creating tensors of the same attributes using `param.new_*`.
        """
        return next(self.parameters())

    def init_batch(self):
        r"""Perform batch-specific initialization routines. For most cells this
        is a no-op.
        """
        pass

    def zero_state(self, batch_size: int) -> State:
        r"""Return zero-filled state tensor(s).

        Args:
            batch_size: int, the batch size.

        Returns:
            State tensor(s) initialized to zeros. Note that different subclasses
            might return tensors of different shapes and structures.
        """
        self.init_batch()
        if isinstance(self._cell, nn.RNNCellBase):
            state = self._param.new_zeros(
                batch_size, self.hidden_size, requires_grad=False)
        else:
            state = self._cell.zero_state(batch_size)
        return state

    def forward(self,  # type: ignore
                input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        r"""
        Returns:
            A tuple of (output, state). For single layer RNNs, output is
            the same as state.
        """
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        return self._cell(input, state)


class BuiltinCellWrapper(RNNCellBase[State]):
    r"""Base class for wrappers over built-in :torch_nn:`RNNCellBase`
    RNN cells.
    """

    def forward(self,  # type: ignore
                input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        new_state = self._cell(input, state)
        return new_state, new_state


class RNNCell(BuiltinCellWrapper[RNNState]):
    r"""A wrapper over :torch_nn:`RNNCell`."""

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        cell = nn.RNNCell(
            input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)
        super().__init__(cell)


class GRUCell(BuiltinCellWrapper[RNNState]):
    r"""A wrapper over :torch_nn:`GRUCell`."""

    def __init__(self, input_size, hidden_size, bias=True):
        cell = nn.GRUCell(input_size, hidden_size, bias=bias)
        super().__init__(cell)


class LSTMCell(BuiltinCellWrapper[LSTMState]):
    r"""A wrapper over :torch_nn:`LSTMCell`, additionally providing the
    option to initialize the forget-gate bias to a constant value.
    """

    def __init__(self, input_size, hidden_size, bias=True,
                 forget_bias: Optional[float] = None):
        if forget_bias is not None and not bias:
            raise ValueError("Parameter 'forget_bias' must be set to None when"
                             "'bias' is set to False.")
        cell = nn.LSTMCell(input_size, hidden_size, bias=bias)
        if forget_bias is not None:
            with torch.no_grad():
                cell.bias_ih[hidden_size:(2 * hidden_size)].fill_(forget_bias)
                cell.bias_hh[hidden_size:(2 * hidden_size)].fill_(forget_bias)
        super().__init__(cell)

    def zero_state(self, batch_size: int) -> LSTMState:
        r"""Returns the zero state for LSTMs as (h, c)."""
        state = self._param.new_zeros(
            batch_size, self.hidden_size, requires_grad=False)
        return state, state

    def forward(self,  # type: ignore
                input: torch.Tensor, state: Optional[LSTMState] = None) \
            -> Tuple[torch.Tensor, LSTMState]:
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        new_state = self._cell(input, state)
        return new_state[0], new_state


class DropoutWrapper(RNNCellBase[State]):
    r"""Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell: RNNCellBase[State],
                 input_keep_prob: float = 1.0,
                 output_keep_prob: float = 1.0,
                 state_keep_prob: float = 1.0,
                 variational_recurrent=False):
        r"""Create a cell with added input, state, and/or output dropout.

        If `variational_recurrent` is set to `True` (**NOT** the default
        behavior), then the same dropout mask is applied at every step, as
        described in:

        Y. Gal, Z Ghahramani.  "A Theoretically Grounded Application of Dropout
        in Recurrent Neural Networks".  https://arxiv.org/abs/1512.05287

        Otherwise a different dropout mask is applied at every time step.

        Note, by default (unless a custom `dropout_state_filter` is provided),
        the memory state (`c` component of any `LSTMStateTuple`) passing through
        a `DropoutWrapper` is never modified.  This behavior is described in the
        above article.

        Args:
            cell: an RNNCell.
            input_keep_prob: float between 0 and 1, input keep probability;
                if it is constant and 1, no input dropout will be added.
            output_keep_prob: float between 0 and 1, output keep probability;
                if it is constant and 1, no output dropout will be added.
            state_keep_prob: float between 0 and 1, output keep probability;
                if it is constant and 1, no output dropout will be added.
                State dropout is performed on the outgoing states of the cell.
            variational_recurrent: bool.  If `True`, then the same dropout
                pattern is applied across all time steps for one batch. This is
                implemented by initializing dropout masks in :meth:`zero_state`.
        """
        super().__init__(cell)

        for prob, attr in [(input_keep_prob, "input_keep_prob"),
                           (state_keep_prob, "state_keep_prob"),
                           (output_keep_prob, "output_keep_prob")]:
            if prob < 0.0 or prob > 1.0:
                raise ValueError(
                    f"Parameter '{attr}' must be between 0 and 1: {prob:d}")

        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._state_keep_prob = state_keep_prob

        self._variational_recurrent = variational_recurrent
        self._recurrent_input_mask: Optional[torch.Tensor] = None
        self._recurrent_output_mask: Optional[torch.Tensor] = None
        self._recurrent_state_mask: Optional[torch.Tensor] = None

    def _new_mask(self, batch_size: int, mask_size: int,
                  prob: float) -> torch.Tensor:
        return self._param.new_zeros(batch_size, mask_size).bernoulli_(prob)

    def init_batch(self):
        r"""Initialize dropout masks for variational dropout.

        Note that we do not create dropout mask here, because the batch size
        may not be known until actual input is passed in.
        """
        self._recurrent_input_mask = None
        self._recurrent_output_mask = None
        self._recurrent_state_mask = None

    def _dropout(self, tensor: torch.Tensor, keep_prob: float,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Decides whether to perform standard dropout or recurrent dropout."""
        if keep_prob == 1.0 or not self.training:
            return tensor
        if mask is not None:
            return tensor.mul(mask).mul(1.0 / keep_prob)
        return F.dropout(tensor, 1.0 - keep_prob, self.training)

    def forward(self,  # type: ignore
                input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        if self.training and self._variational_recurrent:
            # Create or check recurrent masks.
            batch_size = input.size(0)
            for name, size in [('input', self.input_size),
                               ('output', self.hidden_size),
                               ('state', self.hidden_size)]:
                prob = getattr(self, f'_{name}_keep_prob')
                if prob == 1.0:
                    continue
                mask = getattr(self, f'_recurrent_{name}_mask')
                if mask is None:
                    # Initialize the mask according to current batch size.
                    mask = self._new_mask(batch_size, size, prob)
                    setattr(self, f'_recurrent_{name}_mask', mask)
                else:
                    # Check that size matches.
                    if mask.size(0) != batch_size:
                        raise ValueError(
                            "Variational recurrent dropout mask does not "
                            "support variable batch sizes across time steps")

        input = self._dropout(input, self._input_keep_prob,
                              self._recurrent_input_mask)
        output, new_state = super().forward(input, state)
        output = self._dropout(output, self._output_keep_prob,
                               self._recurrent_output_mask)
        new_state = utils.map_structure(
            lambda x: self._dropout(
                x, self._state_keep_prob, self._recurrent_state_mask),
            new_state)
        return output, new_state


class ResidualWrapper(RNNCellBase[State]):
    r"""RNNCell wrapper that ensures cell inputs are added to the outputs."""

    def forward(self,  # type: ignore
                input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        output, new_state = super().forward(input, state)
        output = input + output
        return output, new_state


class HighwayWrapper(RNNCellBase[State]):
    r"""RNNCell wrapper that adds highway connection on cell input and output.

    Based on: `R. K. Srivastava, K. Greff, and J. Schmidhuber, "Highway
    networks", arXiv preprint arXiv:1505.00387, 2015.`
    https://arxiv.org/pdf/1505.00387.pdf
    """

    def __init__(self, cell: RNNCellBase[State],
                 carry_bias_init: Optional[float] = None,
                 couple_carry_transform_gates: bool = True):
        r"""Constructs a `HighwayWrapper` for `cell`.

        Args:
            cell: An instance of `RNNCell`.
            carry_bias_init: float, carry gates bias initialization.
            couple_carry_transform_gates: boolean, should the Carry and
                Transform gate be coupled.
        """
        super().__init__(cell)

        self.carry = nn.Linear(self.input_size, self.input_size)
        if not couple_carry_transform_gates:
            self.transform = nn.Linear(self.input_size, self.input_size)

        self._coupled = couple_carry_transform_gates
        if carry_bias_init is not None:
            nn.init.constant_(self.carry.bias, carry_bias_init)
            if not couple_carry_transform_gates:
                nn.init.constant_(self.transform.bias, -carry_bias_init)

    def forward(self,  # type: ignore
                input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        output, new_state = super().forward(input, state)
        carry = torch.sigmoid(self.carry(input))
        if self._coupled:
            transform = 1 - carry
        else:
            transform = torch.sigmoid(self.transform(input))
        output = input * carry + output * transform
        return output, new_state


class MultiRNNCell(RNNCellBase[List[State]]):
    r"""RNN cell composed sequentially of multiple simple cells.

    .. code-block:: python

        sizes = [128, 128, 64]
        cells = [BasicLSTMCell(input_size, hidden_size)
                 for input_size, hidden_size in zip(sizes[:-1], sizes[1:])]
        stacked_rnn_cell = MultiRNNCell(cells)
    """

    _cell: nn.ModuleList  # type: ignore

    def __init__(self, cells: List[RNNCellBase[State]]):
        r"""Create a RNN cell composed sequentially of a number of RNNCells.

        Args:
          cells: list of RNNCells that will be composed in this order.

        Raises:
          ValueError: if cells is empty (not allowed).
        """
        if len(cells) == 0:
            raise ValueError("Parameter 'cells' should not be empty.")
        cell = nn.ModuleList(cells)
        super().__init__(cell)  # type: ignore

    @property
    def input_size(self):
        return self._cell[0].input_size

    @property
    def hidden_size(self):
        return self._cell[-1].hidden_size

    def init_batch(self):
        for cell in self._cell:
            cell.init_batch()

    def zero_state(self, batch_size: int) -> List[State]:
        states = [cell.zero_state(batch_size)  # type: ignore
                  for cell in self._cell]
        return states

    def forward(self,  # type: ignore
                input: torch.Tensor,
                state: Optional[List[State]] = None) \
            -> Tuple[torch.Tensor, List[State]]:
        r"""Run this multi-layer cell on inputs, starting from state."""
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        new_states = []
        output = input
        for cell, hx in zip(self._cell, state):
            output, new_state = cell(output, hx)
            new_states.append(new_state)
        return output, new_states


class AttentionWrapper(RNNCellBase[AttentionWrapperState]):
    r"""Wraps another `RNNCell` with attention."""

    def __init__(self,
                 cell: RNNCellBase,
                 attention_mechanism: MaybeList[AttentionMechanism],
                 attention_layer_size: Optional[MaybeList[int]] = None,
                 alignment_history: bool = False,
                 cell_input_fn: Optional[Callable[[torch.Tensor, torch.Tensor],
                                                  torch.Tensor]] = None,
                 output_attention: bool = True):
        r"""Wraps RNN cell with attention.
        Construct the `AttentionWrapper`.

        Args:
            cell: An instance of RNN cell.
            attention_mechanism: A list of
                :class:`~texar.torch.core.AttentionMechanism` instances or a
                single instance.
            attention_layer_size: A list of Python integers or a single Python
                integer, the depth of the attention (output) layer(s). If None
                (default), use the context as attention at each time step.
                Otherwise, feed the context and cell output into the attention
                layer to generate attention at each time step. If
                attention_mechanism is a list, attention_layer_size must be a
                list of the same length.
            alignment_history (bool): whether to store alignment
                history from all time steps in the final output state.
            cell_input_fn (optional): A `callable`.  The default is:
                `lambda inputs, attention: array_ops.concat([inputs, attention],
                -1)`.
            output_attention (bool): If `True` (default), the output at
                each time step is the attention value.  This is the behavior of
                Luong-style attention mechanisms. If `False`, the output at
                each time step is the output of `cell`.  This is the behavior
                of Bahdanau-style attention mechanisms.  In both cases, the
                `attention` tensor is propagated to the next time step via the
                state and is used there. This flag only controls whether the
                attention mechanism is propagated up to the next cell in an RNN
                stack or to the top RNN output.

        Raises:
            TypeError: :attr:`attention_layer_size` is not None and
                `attention_mechanism` is a list but
                :attr:`attention_layer_size` is not; or vice versa.
            ValueError: if `attention_layer_size` is not None,
                :attr:`attention_mechanism` is a list, and its length does not
                match that of :attr:`attention_layer_size`; if
                :attr:`attention_layer_size` and `attention_layer` are set
                simultaneously.
        """
        super().__init__(cell)

        self._is_multi: bool
        if isinstance(attention_mechanism, (list, tuple)):
            self._is_multi = True
            attention_mechanisms = attention_mechanism
            for mechanism in attention_mechanisms:
                if not isinstance(mechanism, AttentionMechanism):
                    raise TypeError(
                        "attention_mechanism must contain only instances of "
                        "AttentionMechanism, saw type: %s" %
                        type(mechanism).__name__)
        else:
            self._is_multi = False
            if not isinstance(attention_mechanism, AttentionMechanism):
                raise TypeError(
                    "attention_mechanism must be an AttentionMechanism or list "
                    "of multiple AttentionMechanism instances, saw type: %s" %
                    type(attention_mechanism).__name__)
            attention_mechanisms = [attention_mechanism]

        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, attention: torch.cat((inputs, attention),
                                                    dim=-1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                    "cell_input_fn must be callable, saw type: %s" %
                    type(cell_input_fn).__name__)

        self._attention_layers: Optional[nn.ModuleList]

        if attention_layer_size is not None:
            if isinstance(attention_layer_size, (list, tuple)):
                attention_layer_sizes = tuple(attention_layer_size)
            else:
                attention_layer_sizes = (attention_layer_size,)

            if len(attention_layer_sizes) != len(attention_mechanisms):
                raise ValueError(
                    "If provided, attention_layer_size must contain exactly "
                    "one integer per attention_mechanism, saw: %d vs %d"
                    % (len(attention_layer_sizes), len(attention_mechanisms)))

            self._attention_layers = nn.ModuleList(
                nn.Linear(attention_mechanisms[i].encoder_output_size +
                          cell.hidden_size,
                          attention_layer_sizes[i],
                          False) for i in range(len(attention_layer_sizes)))
            self._attention_layer_size = sum(attention_layer_sizes)
        else:
            self._attention_layers = None
            self._attention_layer_size = sum(
                attention_mechanism.encoder_output_size
                for attention_mechanism in attention_mechanisms)

        self._cell = cell
        self.attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        self._initial_cell_state = None

    def _item_or_tuple(self, seq):
        r"""Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the AttentionMechanism(s) were
        passed to the constructor.

        Args:
            seq: A non-empty sequence of items or generator.

        Returns:
            Either the values in the sequence as a tuple if
            AttentionMechanism(s) were passed to the constructor as a sequence
            or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    @property
    def output_size(self) -> int:
        r"""The number of features in the output tensor."""
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.hidden_size

    def zero_state(self,
                   batch_size: int) -> AttentionWrapperState:
        r"""Return an initial (zero) state tuple for this
        :class:`AttentionWrapper`.

        .. note::
                Please see the initializer documentation for details of how
                to call :meth:`zero_state` if using an
                :class:`~texar.torch.core.AttentionWrapper` with a
                :class:`~texar.torch.modules.BeamSearchDecoder`.

        Args:
            batch_size: `0D` integer: the batch size.

        Returns:
            An :class:`~texar.torch.core.AttentionWrapperState` tuple containing
            zeroed out tensors and Python lists.
        """
        cell_state: torch.Tensor = super().zero_state(batch_size)  # type:ignore

        initial_alignments = [None for _ in self.attention_mechanisms]

        alignment_history: List[List[Optional[torch.Tensor]]]
        alignment_history = [[] for _ in initial_alignments]

        return AttentionWrapperState(
            cell_state=cell_state,
            time=0,
            attention=self._param.new_zeros(batch_size,
                                            self._attention_layer_size,
                                            requires_grad=False),
            alignments=self._item_or_tuple(initial_alignments),
            attention_state=self._item_or_tuple(initial_alignments),
            alignment_history=self._item_or_tuple(alignment_history))

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                state: Optional[AttentionWrapperState],
                memory: torch.Tensor,
                memory_sequence_length: Optional[torch.LongTensor] = None) -> \
            Tuple[torch.Tensor, AttentionWrapperState]:
        r"""Perform a step of attention-wrapped RNN.

        - Step 1: Mix the :attr:`inputs` and previous step's `attention` output
          via `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous
          state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell
          output and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).

        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time
                step.
            state: An instance of
                :class:`~texar.torch.core.AttentionWrapperState` containing
                tensors from the previous time step.
            memory: The memory to query; usually the output of an RNN encoder.
                This tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length: (optional) Sequence lengths for the batch
                entries in memory.  If provided, the memory tensor rows are
                masked with zeros for values past the respective sequence
                lengths.

        Returns:
            A tuple `(attention_or_cell_output, next_state)`, where

            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of
              :class:`~texar.torch.core.AttentionWrapperState` containing the
              state calculated at this time step.

        Raises:
            TypeError: If `state` is not an instance of
                :class:`~texar.torch.core.AttentionWrapperState`.
        """
        if state is None:
            state = self.zero_state(batch_size=memory.shape[0])
        elif not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of "
                            "AttentionWrapperState. Received type %s instead."
                            % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state

        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]  # type: ignore
            previous_alignment_history = \
                [state.alignment_history]  # type: ignore

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self.attention_mechanisms):
            if previous_attention_state[i] is not None:
                attention_state = previous_attention_state[i]
            else:
                attention_state = attention_mechanism.initial_state(
                    memory.shape[0], memory.shape[1], self._param.dtype,
                    self._param.device)

            attention, alignments, next_attention_state = compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=attention_state,
                attention_layer=(self._attention_layers[i]
                                 if self._attention_layers else None),
                memory=memory,
                memory_sequence_length=memory_sequence_length)

            if self._alignment_history:
                alignment_history = previous_alignment_history[i] + [alignments]
            else:
                alignment_history = previous_alignment_history[i]

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = torch.cat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state
