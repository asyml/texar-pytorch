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
"""RNN helpers for PyTorch models."""

from typing import List, Optional, Tuple, TypeVar, Union

import torch

from texar.torch.core.cell_wrappers import RNNCellBase
from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.utils import map_structure, map_structure_zip, no_map

__all__ = [
    "reverse_sequence",
    "dynamic_rnn",
    "bidirectional_dynamic_rnn",
]

State = TypeVar('State')


def reverse_sequence(inputs: torch.Tensor,
                     seq_lengths: Union[torch.LongTensor, List[int]],
                     time_major: bool) -> torch.Tensor:
    r"""Reverses variable length slices.

    This op first slices input along the dimension batch_axis, and for each
    slice i, reverses the first seq_lengths[i] elements along the dimension
    seq_axis.

    The elements of seq_lengths must obey seq_lengths[i] <=
    input.dims[seq_dim], and seq_lengths must be a vector of length
    input.dims[batch_dim].

    The output slice i along dimension batch_axis is then given by input slice
    i, with the first seq_lengths[i] slices along dimension seq_axis reversed.

    Args:
        inputs: A Tensor. The input to reverse.
        seq_lengths: A Tensor. Must be one of the following types: int32,
            int64. 1-D with length input.dims(batch_dim) and
            max(seq_lengths) <= input.dims(seq_dim)
        time_major: The shape format of the ``inputs`` and ``outputs`` Tensors.
            If true, these ``Tensors`` must be shaped
            ``[max_time, batch_size, depth]``. If false, these ``Tensors`` must
            be shaped ``[batch_size, max_time, depth]``.
            Using ``time_major = True`` is a bit more efficient because it
            avoids transposes at the beginning and end of the RNN calculation.
            However, most TensorFlow data is batch-major, so by
            default this functionb accepts input and emits output
            in batch-major form.

    Returns:
        A ``Tensor``. Has the same type as input.
    """
    if time_major:
        inputs = inputs.permute(1, 0, 2)

    batch_size = inputs.shape[0]

    outputs = inputs.clone()
    for i in range(batch_size):
        outputs[i][0:seq_lengths[i]] = torch.flip(
            inputs[i][0:seq_lengths[i]], dims=(0,))
    if time_major:
        outputs = outputs.permute(1, 0, 2)

    return outputs


def bidirectional_dynamic_rnn(
        cell_fw: RNNCellBase[State],
        cell_bw: RNNCellBase[State],
        inputs: torch.Tensor,
        sequence_length: Optional[Union[torch.LongTensor, List[int]]] = None,
        initial_state_fw: Optional[State] = None,
        initial_state_bw: Optional[State] = None,
        time_major: bool = False) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                           Tuple[State, State]]:
    r"""Creates a dynamic version of bidirectional recurrent neural network.

    Takes input and builds independent forward and backward RNNs. The
    input_size of forward and backward cell must match. The initial state for
    both directions is zero by default (but can be set optionally) and no
    intermediate states are ever returned -- the network is fully unrolled
    for the given (passed in) length(s) of the sequence(s) or completely
    unrolled if length(s) is not given.

    Args:
        cell_fw: An instance of RNNCell, to be used for forward direction.
        cell_bw: An instance of RNNCell, to be used for backward direction.
        inputs: The RNN inputs.
            If time_major == False (default), this must be a tensor of shape:
            ``[batch_size, max_time, ...]``, or a nested tuple of such elements.
            If time_major == True, this must be a tensor of shape:
            ``[max_time, batch_size, ...]``, or a nested tuple of such elements.
        sequence_length: (optional) An int32/int64 tensor, size
            ``[batch_size]``, containing the actual lengths for each of the
            sequences in
            the batch. If not provided, all batch entries are assumed
            to be full sequences; and time reversal is applied from time
            ``0`` to ``max_time`` for each sequence.
        initial_state_fw: (optional) An initial state for the forward RNN.
            This must be a tensor of appropriate type and shape
            ``[batch_size, cell_fw.state_size]``.
            If ``cell_fw.state_size`` is a tuple, this should be a tuple of
            tensors having shapes ``[batch_size, s]``
            for ``s`` in ``cell_fw.state_size``.
        initial_state_bw: (optional) Same as for ``initial_state_fw``, but using
            the corresponding properties of ``cell_bw``.
        time_major: The shape format of the ``inputs`` and ``outputs`` Tensors.
            If true, these ``Tensors`` must be shaped
            ``[max_time, batch_size, depth]``.
            If false, these ``Tensors`` must be shaped
            ``[batch_size, max_time, depth]``.
            Using ``time_major = True`` is a bit more efficient because it
            avoids transposes at the beginning and end of the RNN calculation.
            However, most TensorFlow data is batch-major, so by
            default this function accepts input and emits output
            in batch-major form.

    Returns:
        A tuple (outputs, output_states) where:

        outputs: A tuple (output_fw, output_bw) containing the forward and
            the backward rnn output ``Tensor``.
            If time_major == False (default),
                output_fw will be a ``Tensor`` shaped:
                ``[batch_size, max_time, cell_fw.output_size]``
                and output_bw will be a ``Tensor`` shaped:
                ``[batch_size, max_time, cell_bw.output_size]``.
            If time_major == True,
                output_fw will be a ``Tensor`` shaped:
                ``[max_time, batch_size, cell_fw.output_size]``
                and output_bw will be a ``Tensor`` shaped:
                ``[max_time, batch_size, cell_bw.output_size]``.
            It returns a tuple instead of a single concatenated ``Tensor``,
            unlike in the ``bidirectional_rnn``. If the concatenated
            one is preferred, the forward and backward outputs can
            be concatenated as ``tf.concat(outputs, 2)``.
        output_states: A tuple (output_state_fw, output_state_bw) containing
            the forward and the backward final states of bidirectional rnn.
    """
    output_fw, output_state_fw = dynamic_rnn(cell=cell_fw,
                                             inputs=inputs,
                                             sequence_length=sequence_length,
                                             initial_state=initial_state_fw,
                                             time_major=time_major)
    if time_major:
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
    else:
        time_steps = inputs.shape[1]
        batch_size = inputs.shape[0]

    if sequence_length is None:
        sequence_length = torch.tensor([time_steps] * batch_size,
                                       dtype=torch.int32,
                                       device=inputs.device)

    # Backward direction
    inputs_reverse = reverse_sequence(inputs=inputs,
                                      seq_lengths=sequence_length,
                                      time_major=time_major)

    tmp, output_state_bw = dynamic_rnn(cell=cell_bw,
                                       inputs=inputs_reverse,
                                       sequence_length=sequence_length,
                                       initial_state=initial_state_bw,
                                       time_major=time_major)
    output_bw = reverse_sequence(inputs=tmp,
                                 seq_lengths=sequence_length,
                                 time_major=time_major)

    outputs = (output_fw, output_bw)
    output_states = (output_state_fw, output_state_bw)

    return outputs, output_states


def dynamic_rnn(
        cell: RNNCellBase[State],
        inputs: torch.Tensor,
        sequence_length: Optional[Union[torch.LongTensor, List[int]]] = None,
        initial_state: Optional[State] = None,
        time_major: bool = False) -> Tuple[torch.Tensor, State]:
    r"""Creates a recurrent neural network specified by RNNCell ``cell``.

    Performs fully dynamic unrolling of ``inputs``.

    Args:
        cell: An instance of RNNCell.
        inputs: The RNN inputs.
            If ``time_major == False`` (default), this must be a ``Tensor``
            of shape: ``[batch_size, max_time, ...]``, or a nested
            tuple of such elements.
            If ``time_major == True``, this must be a ``Tensor`` of shape:
            ``[max_time, batch_size, ...]``, or a nested tuple of such
            elements.
            This may also be a (possibly nested) tuple of Tensors satisfying
            this property.  The first two dimensions must match across all the
            inputs, but otherwise the ranks and other shape components
            may differ. In this case, input to ``cell`` at each time-step
            will replicate the structure of these tuples, except for the
            time dimension (from which the time is taken).
            The input to ``cell`` at each time step will be a
            ``Tensor`` or (possibly nested) tuple of Tensors each with
            dimensions ``[batch_size, ...]``.
        sequence_length: (optional) An int32/int64 tensor sized
            ``[batch_size]``. Used to copy-through state and
            zero-out outputs when past a batch element's sequence length.
            So it's more for performance than correctness.
        initial_state: (optional) An initial state for the RNN.
            If ``cell.state_size`` is an integer, this must be
            a ``Tensor`` of appropriate type and shape
            ``[batch_size, cell.state_size]``. If ``cell.state_size`` is
            a tuple, this should be a tuple of tensors having shapes
            ``[batch_size, s]`` for ``s`` in ``cell.state_size``.
        time_major: The shape format of the ``inputs`` and ``outputs`` Tensors.
            If true, these ``Tensors`` must be shaped
            ``[max_time, batch_size, depth]``. If false, these ``Tensors``
            must be shaped ``[batch_size, max_time, depth]``.
            Using ``time_major = True`` is a bit more efficient because
            it avoids transposes at the beginning and end of the
            RNN calculation. However, most TensorFlow data is batch-major,
            so by default this function accepts input and emits output in
            batch-major form.

    Returns:
        A pair (outputs, state) where:

        outputs: The RNN output ``Tensor``.

            If time_major == False (default), this will be a ``Tensor`` shaped:
            ``[batch_size, max_time, cell.output_size]``.

            If time_major == True, this will be a ``Tensor`` shaped:
            ``[max_time, batch_size, cell.output_size]``.

            Note, if ``cell.output_size`` is a (possibly nested) tuple of
            integers or ``torch.Size`` objects, then ``outputs``
            will be a tuple having the same structure as ``cell.output_size``,
            containing Tensors having shapes corresponding to the shape
            data in ``cell.output_size``.

        state: The final state.  If ``cell.state_size`` is an int, this
            will be shaped ``[batch_size, cell.state_size]``.  If it is a
            ``torch.Size``, this will be shaped
            ``[batch_size] + cell.state_size``.
            If it is a (possibly nested) tuple of ints or ``torch.Size``,
            this will be a tuple having the corresponding shapes.
            If cells are ``LSTMCells``, ``state`` will be a tuple containing
            a ``LSTMStateTuple`` for each cell.

    Raises:
        TypeError: If ``cell`` is not an instance of RNNCell.
        ValueError: If inputs is None or an empty list.
    """
    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    if not time_major:
        # (B,T,D) => (T,B,D)
        inputs = inputs.permute(1, 0, 2)

    time_steps = inputs.shape[0]
    batch_size = inputs.shape[1]

    if sequence_length is not None:
        if not isinstance(sequence_length, torch.Tensor):
            sequence_length = torch.tensor(sequence_length,
                                           dtype=torch.int32,
                                           device=inputs.device)

        if sequence_length.dim() != 1:
            raise ValueError(
                "sequence_length must be a vector of length batch_size, "
                "but saw shape: %s" % sequence_length.shape)
        if sequence_length.shape != torch.Size([batch_size]):
            raise ValueError("Expected shape for Tensor sequence_length is %s"
                             % batch_size, " but saw shape: %s"
                             % sequence_length.shape)
    else:
        sequence_length = torch.tensor([time_steps] * batch_size,
                                       dtype=torch.int32,
                                       device=inputs.device)

    if initial_state is not None:
        state = initial_state
    else:
        state = cell.zero_state(batch_size=batch_size)

    (outputs, final_state) = _dynamic_rnn_loop(
        cell, inputs, state, sequence_length=sequence_length)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
        # (T,B,D) => (B,T,D)
        outputs = outputs.permute(1, 0, 2)

    return outputs, final_state


def _dynamic_rnn_loop(cell: RNNCellBase[State],
                      inputs: torch.Tensor,
                      initial_state: State,
                      sequence_length: torch.LongTensor) \
        -> Tuple[torch.Tensor, State]:
    r"""Internal implementation of Dynamic RNN.

    Args:
        cell: An instance of RNNCell.
        inputs: A ``Tensor`` of shape ``[time, batch_size, input_size]``,
            or a nested tuple of such elements.
        initial_state: A ``Tensor`` of shape ``[batch_size, state_size]``,
            or if ``cell.state_size`` is a tuple, then this should be a tuple
            of tensors having shapes ``[batch_size, s]`` for ``s`` in
            ``cell.state_size``.
        sequence_length: (optional) An ``int32`` ``Tensor``
            of shape ``[batch_size]``.

    Returns:
        Tuple ``(final_outputs, final_state)``.
        final_outputs:
            A ``Tensor`` of shape ``[time, batch_size, cell.output_size]``. If
            ``cell.output_size`` is a (possibly nested) tuple of ints or
            ``torch.Size`` objects, then this returns a
            (possibly nested) tuple of Tensors matching the corresponding
            shapes.
        final_state:
            A ``Tensor``, or possibly nested tuple of Tensors, matching
            in length and shapes to ``initial_state``.
    """
    state = initial_state
    time_steps = inputs.shape[0]
    all_outputs = []

    all_state = map_structure(lambda _: no_map(list), state)

    for i in range(time_steps):
        output, state = cell(inputs[i], state)
        all_outputs.append(output)
        map_structure_zip(lambda xs, x: xs.append(x), (all_state, state))
    # TODO: Do not compute everything regardless of sequence_length

    final_outputs = torch.stack(all_outputs, dim=0)
    final_outputs = mask_sequences(final_outputs,
                                   sequence_length=sequence_length,
                                   time_major=True)

    final_state = map_structure(lambda _: no_map(list), state)
    # pylint: disable=cell-var-from-loop
    # Our use case is fine because the function is called immediately and
    # exclusively in the current iteration of the loop.
    for batch_idx, time_idx in enumerate(sequence_length.tolist()):
        if time_idx > 0:
            map_structure_zip(
                lambda xs, x: xs.append(x[time_idx - 1][batch_idx]),
                (final_state, all_state))
        else:
            map_structure_zip(
                lambda xs, x: xs.append(x[batch_idx]),
                (final_state, initial_state))
    # pylint: enable=cell-var-from-loop

    final_state = map_structure(
            lambda x: torch.stack(x, dim=0), final_state)

    return final_outputs, final_state
