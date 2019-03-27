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

from typing import Optional, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from texar.utils import utils

__all__ = [
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

State = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def wrap_builtin_cell(cell: nn.RNNCellBase):
    # convert cls to corresponding derived wrapper class
    if isinstance(cell, nn.RNNCell):
        self = RNNCellBase.__new__(RNNCell)
    elif isinstance(cell, nn.GRUCell):
        self = RNNCellBase.__new__(GRUCell)
    elif isinstance(cell, nn.LSTMCell):
        self = RNNCellBase.__new__(LSTMCell)
    else:
        raise TypeError(f"Unrecognized class {type(cell)}.")
    self._cell = cell
    return self


class RNNCellBase(nn.Module):
    """The base class for RNN cells in our framework. Major differences over
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
        return self._cell.input_size

    @property
    def hidden_size(self):
        return self._cell.hidden_size

    @property
    def _param(self) -> nn.Parameter:
        return next(self.parameters())

    def zero_state(self, batch_size: int):
        if isinstance(self._cell, nn.RNNCellBase):
            state = self._param.new_zeros(
                batch_size, self.hidden_size, requires_grad=False)
        else:
            state = self._cell.zero_state(batch_size)
        return state

    def forward(self, input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        """
        :return: A tuple of (output, state). For single layer RNNs, output is
            the same as state.
        """
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        return self._cell(input, state)


class BuiltinCellWrapper(RNNCellBase):
    def forward(self, input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        """
        :return: A tuple of (output, state). For single layer RNNs, output is
            the same as state.
        """
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        new_state = self._cell(input, state)
        return new_state, new_state


class RNNCell(BuiltinCellWrapper):
    """A wrapper over :class:`torch.nn.RNNCell`.
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        cell = nn.RNNCell(
            input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)
        super().__init__(cell)


class GRUCell(BuiltinCellWrapper):
    """A wrapper over :class:`torch.nn.GRUCell`.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        cell = nn.GRUCell(input_size, hidden_size, bias=bias)
        super().__init__(cell)


class LSTMCell(BuiltinCellWrapper):
    """A wrapper over :class:`torch.nn.LSTMCell`, additionally providing the
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

    def zero_state(self, batch_size: int):
        state = super().zero_state(batch_size)
        return (state, state)

    def forward(self, input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        new_state = self._cell(input, state)
        return new_state[0], new_state


class DropoutWrapper(RNNCellBase):
    """Create a cell with added input, state, and/or output dropout.

    If `variational_recurrent` is set to `True` (**NOT** the default behavior),
    then the same dropout mask is applied at every step, as described in:

    Y. Gal, Z Ghahramani.  "A Theoretically Grounded Application of Dropout in
    Recurrent Neural Networks".  https://arxiv.org/abs/1512.05287

    Otherwise a different dropout mask is applied at every time step.

    Note, by default (unless a custom `dropout_state_filter` is provided),
    the memory state (`c` component of any `LSTMStateTuple`) passing through
    a `DropoutWrapper` is never modified.  This behavior is described in the
    above article.

    Args:
      cell: an RNNCell.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is constant and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
      state_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
        State dropout is performed on the outgoing states of the cell.
      variational_recurrent: Python bool.  If `True`, then the same
        dropout pattern is applied across all time steps for one batch. This
        is implemented by initializing dropout masks in :meth:`zero_state`.
    """

    def __init__(self, cell: nn.Module,
                 input_keep_prob: float = 1.0,
                 output_keep_prob: float = 1.0,
                 state_keep_prob: float = 1.0,
                 variational_recurrent=False):
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
        self._recurrent_input_mask = None
        self._recurrent_output_mask = None
        self._recurrent_state_mask = None

    def zero_state(self, batch_size: int):
        state = super().zero_state(batch_size)
        self._recurrent_input_mask = None
        self._recurrent_output_mask = None
        self._recurrent_state_mask = None
        # pylint: disable=line-too-long
        if self.training and self._variational_recurrent:
            if self._input_keep_prob < 1.0:
                self._recurrent_input_mask = self._param.new_zeros(
                    batch_size, self.input_size).bernoulli_(self._input_keep_prob)
            if self._output_keep_prob < 1.0:
                self._recurrent_output_mask = self._param.new_zeros(
                    batch_size, self.hidden_size).bernoulli_(self._output_keep_prob)
            if self._state_keep_prob < 1.0:
                self._recurrent_state_mask = self._param.new_zeros(
                    batch_size, self.hidden_size).bernoulli_(self._state_keep_prob)
        # pylint: enable=line-too-long
        return state

    def _dropout(self, tensor: torch.Tensor, keep_prob: float,
                 mask: Optional[torch.Tensor] = None):
        if keep_prob == 1.0:
            return tensor
        if mask is not None:
            return tensor.mul(mask).mul(1.0 / keep_prob)
        return F.dropout(tensor, 1.0 - keep_prob, self.training)

    def forward(self, input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
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


class ResidualWrapper(RNNCellBase):
    def __init__(self, cell: nn.Module):
        super().__init__(cell)

    def forward(self, input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        output, new_state = super().forward(input, state)
        output = input + output
        return output, new_state


class HighwayWrapper(RNNCellBase):
    def __init__(self, cell: nn.Module, carry_bias_init: Optional[float] = None,
                 couple_carry_transform_gates=True):
        super().__init__(cell)

        self.carry = nn.Linear(self.input_size, self.input_size)
        if not couple_carry_transform_gates:
            self.transform = nn.Linear(self.input_size, self.input_size)

        self._coupled = couple_carry_transform_gates
        if carry_bias_init is not None:
            nn.init.constant_(self.carry.bias, carry_bias_init)
            if not couple_carry_transform_gates:
                nn.init.constant_(self.transform.bias, -carry_bias_init)

    def forward(self, input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        output, new_state = super().forward(input, state)
        carry = torch.sigmoid(self.carry(input))
        if self._coupled:
            transform = 1 - carry
        else:
            transform = torch.sigmoid(self.transform(input))
        output = input * carry + output * transform
        return output, new_state


class MultiRNNCell(RNNCellBase):
    _cell: nn.ModuleList

    def __init__(self, cells: List[nn.Module]):
        cell = nn.ModuleList(cells)
        super().__init__(cell)

    @property
    def input_size(self):
        return self._cell[0].input_size

    @property
    def hidden_size(self):
        return self._cell[-1].hidden_size

    def zero_state(self, batch_size: int):
        states = [cell.zero_state(batch_size) for cell in self._cell]
        return states

    def forward(self, input: torch.Tensor, state: Optional[State] = None) \
            -> Tuple[torch.Tensor, State]:
        if state is None:
            batch_size = input.size(0)
            state = self.zero_state(batch_size)
        new_states = []
        output = input
        for cell, h in zip(self._cell, state):
            output, new_state = cell(output, h)
            new_states.append(new_state)
        return output, new_states
