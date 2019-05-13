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

# pylint: disable=redefined-builtin, arguments-differ, too-many-arguments
# pylint: disable=invalid-name, too-few-public-methods

from typing import Generic, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import nn

from texar.utils import utils
from texar.utils.types import MaybeList

__all__ = [
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
]

State = TypeVar('State')
RNNState = torch.Tensor
LSTMState = Tuple[torch.Tensor, torch.Tensor]

HiddenState = MaybeList[Union[RNNState, LSTMState]]


def wrap_builtin_cell(cell: nn.RNNCellBase):
    r"""Convert a built-in :class:`torch.nn.RNNCellBase` derived RNN cell to
    our wrapped version.

    Args:
        cell: the RNN cell to wrap around.

    Returns:
        The wrapped cell derived from
        :class`texar.core.cell_wrappers.RNNCellBase`.
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
    self._cell = cell  # noqa: E501 pylint: disable=protected-access, attribute-defined-outside-init
    return self


class RNNCellBase(nn.Module, Generic[State]):
    r"""The base class for RNN cells in our framework. Major differences over
    :class:`torch.nn.RNNCell` are two-fold::

        1. Holds an :class:`torch.nn.Module` which could either be a built-in
           RNN cell or a wrapped cell instance. This design allows
           :class:`RNNCellBase` to serve as the base class for both vanilla
           cells and wrapped cells.

        2. Adds :meth:`zero_state` method for initialization of hidden states,
           which can also be used to implement batch-specific initialization
           routines.
    """

    def __init__(self, cell: nn.Module):
        super().__init__()
        if not isinstance(cell, nn.Module):
            raise ValueError("Type of parameter 'cell' must be derived from"
                             "nn.Module, and has 'input_size' and 'hidden_size'"
                             "attributes.")
        self._cell = cell

    @property
    def input_size(self):
        r"""The number of expected features in the input."""
        return self._cell.input_size

    @property
    def hidden_size(self):
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

        Args:
            batch_size: int, the batch size.
        """

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
    r"""Base class for wrappers over built-in :class:`torch.nn.RNNCellBase`
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
    r"""A wrapper over :class:`torch.nn.RNNCell`."""

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        cell = nn.RNNCell(
            input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)
        super().__init__(cell)


class GRUCell(BuiltinCellWrapper[RNNState]):
    r"""A wrapper over :class:`torch.nn.GRUCell`."""

    def __init__(self, input_size, hidden_size, bias=True):
        cell = nn.GRUCell(input_size, hidden_size, bias=bias)
        super().__init__(cell)


class LSTMCell(BuiltinCellWrapper[LSTMState]):
    r"""A wrapper over :class:`torch.nn.LSTMCell`, additionally providing the
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
        return (state, state)

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

    Based on:
        R. K. Srivastava, K. Greff, and J. Schmidhuber, "Highway networks",
        arXiv preprint arXiv:1505.00387, 2015.
        https://arxiv.org/abs/1505.00387
    """

    def __init__(self, cell: nn.Module, carry_bias_init: Optional[float] = None,
                 couple_carry_transform_gates=True):
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
                nn.init.constant_(self.transform.bias, -carry_bias_init)  # noqa: E501 pylint: disable=invalid-unary-operand-type

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

    _cell: nn.ModuleList

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
        super().__init__(cell)

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
        states = [cell.zero_state(batch_size) for cell in self._cell]
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
