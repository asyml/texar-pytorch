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
Various RNN encoders.
"""

from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn

from texar.torch.core import layers
from texar.torch.core.cell_wrappers import LSTMCell, RNNCellBase
from texar.torch.hyperparams import HParams
from texar.torch.modules.encoders.encoder_base import EncoderBase
from texar.torch.modules.networks.conv_networks import _to_list
from texar.torch.utils.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from texar.torch.utils.shapes import mask_sequences

__all__ = [
    "_forward_output_layers",
    "RNNEncoderBase",
    "UnidirectionalRNNEncoder",
    "BidirectionalRNNEncoder",
]

State = TypeVar('State')


def _default_output_layer_hparams() -> Dict[str, Any]:
    return {
        "num_layers": 0,
        "layer_size": 128,
        "activation": "Identity",
        "final_layer_activation": None,
        "other_dense_kwargs": None,
        "dropout_layer_ids": [],
        "dropout_rate": 0.5,
        "variational_dropout": False,
        "@no_typecheck": ["activation", "final_layer_activation",
                          "layer_size", "dropout_layer_ids"]
    }


def _build_dense_output_layer(cell_output_size: int,
                              hparams: HParams) -> Optional[nn.Sequential]:
    r"""Build the output layers.

    Args:
        cell_output_size: The output size of the rnn cell.
        hparams (dict or HParams): Hyperparameters. Missing hyperparameters
            will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    Returns:
        A :torch_nn:`Sequential` module containing the output layers.
    """
    nlayers = hparams.num_layers

    if nlayers <= 0:
        return None

    layer_size = _to_list(
        hparams.layer_size, 'output_layer.layer_size', nlayers)

    dropout_layer_ids = _to_list(hparams.dropout_layer_ids)

    other_kwargs = hparams.other_dense_kwargs or {}
    if isinstance(other_kwargs, HParams):
        other_kwargs = other_kwargs.todict()
    if not isinstance(other_kwargs, dict):
        raise ValueError(
            "hparams 'output_layer.other_dense_kwargs' must be a dict.")

    output_layers: List[nn.Module] = []
    for i in range(nlayers):
        if i in dropout_layer_ids:
            # TODO: Variational dropout is not implemented.
            output_layers.append(nn.Dropout(p=hparams.dropout_rate))

        dense_layer = nn.Linear(in_features=(cell_output_size if i == 0
                                             else layer_size[i - 1]),
                                out_features=layer_size[i], **other_kwargs)

        output_layers.append(dense_layer)

        if i == nlayers - 1:
            activation = hparams.final_layer_activation
        else:
            activation = hparams.activation

        if activation is not None:
            layer_hparams = {"type": activation, "kwargs": {}}
            activation_layer = layers.get_layer(hparams=layer_hparams)
            output_layers.append(activation_layer)

    if nlayers in dropout_layer_ids:
        output_layers.append(nn.Dropout(p=hparams.dropout_rate))

    return nn.Sequential(*output_layers)


def _forward_output_layers(
        inputs: torch.Tensor,
        output_layer: Optional[nn.Module],
        time_major: bool,
        sequence_length: Optional[Union[torch.LongTensor, List[int]]] = None) \
        -> Tuple[torch.Tensor, int]:
    r"""Forwards inputs through the output layers.

    Args:
        inputs: A Tensor of shape ``[batch_size, max_time] + input_size`` if
            :attr:`time_major` is `False`, or shape
            ``[max_time, batch_size] + input_size`` if :attr:`time_major` is
            `True`.
        output_layer (optional): :torch_nn:`Sequential` or :torch_nn:`Module`
            of output layers.
        time_major (bool): The shape format of the :attr:`inputs` and
            :attr:`outputs` Tensors. If `True`, these tensors are of shape
            `[max_time, batch_size, input_size]`. If `False` (default),
            these tensors are of shape `[batch_size, max_time, input_size]`.
        sequence_length (optional): A 1D :tensor:`LongTensor` of shape
            ``[batch_size]``. Sequence lengths of the batch inputs. Used to
            copy-through state and zero-out outputs when past a batch element's
            sequence length.

    Returns:
        A pair :attr:`(outputs, outputs_size), where

        - :attr:`outputs`: A Tensor of shape
        `[batch_size, max_time] + outputs_size`.

        - :attr:`outputs_size`: An `int` representing the output size.
    """
    if output_layer is None:
        return inputs, inputs.shape[-1]

    output = output_layer(inputs)

    if sequence_length is not None:
        output = mask_sequences(output, sequence_length, time_major=time_major)

    output_size = output.shape[-1]

    return output, output_size


class RNNEncoderBase(EncoderBase, Generic[State]):
    r"""Base class for all RNN encoder classes to inherit.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.
    """

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "rnn_encoder"
            }
        """
        return {
            'name': 'rnn_encoder'
        }


class UnidirectionalRNNEncoder(RNNEncoderBase[State]):
    r"""One directional RNN encoder.

    Args:
        cell: (RNNCell, optional) If not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        output_layer (optional): An instance of
            :torch_nn:`Module`. Applies to the RNN cell
            output of each step. If `None` (default), the output layer is
            created as specified in :attr:`hparams["output_layer"]`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    See :meth:`forward` for the inputs and outputs of the encoder.

    Example:

    .. code-block:: python

        # Use with embedder
        embedder = WordEmbedder(vocab_size, hparams=emb_hparams)
        encoder = UnidirectionalRNNEncoder(hparams=enc_hparams)

        outputs, final_state = encoder(
            inputs=embedder(data_batch['text_ids']),
            sequence_length=data_batch['length'])

    .. document private functions
    """
    _cell: RNNCellBase[State]

    def __init__(self,
                 input_size: int,
                 cell: Optional[RNNCellBase[State]] = None,
                 output_layer: Optional[nn.Module] = None,
                 hparams=None):
        super().__init__(hparams=hparams)

        # Make RNN cell
        if cell is not None:
            self._cell = cell
        else:
            self._cell = layers.get_rnn_cell(input_size,
                                             self._hparams.rnn_cell)

        # Make output layer
        self._output_layer: Optional[nn.Module]
        if output_layer is not None:
            self._output_layer = output_layer
            self._output_layer_hparams = None
        else:
            self._output_layer = _build_dense_output_layer(
                self._cell.hidden_size, self._hparams.output_layer)
            self._output_layer_hparams = self._hparams.output_layer

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "rnn_cell": default_rnn_cell_hparams(),
                "output_layer": {
                    "num_layers": 0,
                    "layer_size": 128,
                    "activation": "identity",
                    "final_layer_activation": None,
                    "other_dense_kwargs": None,
                    "dropout_layer_ids": [],
                    "dropout_rate": 0.5,
                    "variational_dropout": False
                },
                "name": "unidirectional_rnn_encoder"
            }

        Here:

        `"rnn_cell"`: dict
            A dictionary of RNN cell hyperparameters. Ignored if
            :attr:`cell` is given to the encoder constructor.

            The default value is defined in
            :func:`~texar.torch.core.default_rnn_cell_hparams`.

        `"output_layer"`: dict
            Output layer hyperparameters. Ignored if :attr:`output_layer`
            is given to the encoder constructor. Includes:

            `"num_layers"`: int
                The number of output (dense) layers. Set to 0 to avoid any
                output layers applied to the cell outputs.

            `"layer_size"`: int or list
                The size of each of the output (dense) layers.

                If an `int`, each output layer will have the same size. If
                a list, the length must equal to :attr:`num_layers`.

            `"activation"`: str or callable or None
                Activation function for each of the output (dense)
                layer except for the final layer. This can be
                a function, or its string name or module path.
                If function name is given, the function must be from
                :mod:`torch.nn`.
                For example:

                .. code-block:: python

                    "activation": "relu" # function name
                    "activation": "my_module.my_activation_fn" # module path
                    "activation": my_module.my_activation_fn # function

                Default is `None` which results in an identity activation.

            `"final_layer_activation"`: str or callable or None
                The activation function for the final output layer.

            `"other_dense_kwargs"`: dict or None
                Other keyword arguments to construct each of the output
                dense layers, e.g., ``bias``. See
                :torch_nn:`Linear` for the keyword arguments.

            `"dropout_layer_ids"`: int or list
                The indexes of layers (starting from 0) whose inputs
                are applied with dropout. The index = :attr:`num_layers`
                means dropout applies to the final layer output. For example,

                .. code-block:: python

                    {
                        "num_layers": 2,
                        "dropout_layer_ids": [0, 2]
                    }

                will leads to a series of layers as
                `-dropout-layer0-layer1-dropout-`.

                The dropout mode (training or not) is controlled
                by :attr:`self.training`.

            `"dropout_rate"`: float
                The dropout rate, between 0 and 1. For example,
                ``"dropout_rate": 0.1`` would zero out 10% of elements.

            `"variational_dropout"`: bool
                Whether the dropout mask is the same across all time steps.

        `"name"`: str
            Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams.update({
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "output_layer": _default_output_layer_hparams(),
            "name": "unidirectional_rnn_encoder"
        })
        return hparams

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                sequence_length: Optional[Union[torch.LongTensor,
                                                List[int]]] = None,
                initial_state: Optional[State] = None,
                time_major: bool = False,
                return_cell_output: bool = False,
                return_output_size: bool = False):
        r"""Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape ``[batch_size, max_time, dim]``.
                The first two dimensions
                :attr:`batch_size` and :attr:`max_time` are exchanged if
                :attr:`time_major` is `True`.
            sequence_length (optional): A 1D :tensor:`LongTensor` of shape
                ``[batch_size]``.
                Sequence lengths of the batch inputs. Used to copy-through
                state and zero-out outputs when past a batch element's sequence
                length.
            initial_state (optional): Initial state of the RNN.
            time_major (bool): The shape format of the :attr:`inputs` and
                :attr:`outputs` Tensors. If `True`, these tensors are of shape
                ``[max_time, batch_size, depth]``. If `False` (default),
                these tensors are of shape ``[batch_size, max_time, depth]``.
            return_cell_output (bool): Whether to return the output of the RNN
                cell. This is the results prior to the output layer.
            return_output_size (bool): Whether to return the size of the
                output (i.e., the results after output layers).

        Returns:
            - By default (both ``return_cell_output`` and ``return_output_size``
              are `False`), returns a pair :attr:`(outputs, final_state)`,
              where

              - :attr:`outputs`: The RNN output tensor by the output layer
                (if exists) or the RNN cell (otherwise). The tensor is of
                shape ``[batch_size, max_time, output_size]`` if
                ``time_major`` is `False`, or
                ``[max_time, batch_size, output_size]`` if
                ``time_major`` is `True`.
                If RNN cell output is a (nested) tuple of Tensors, then the
                :attr:`outputs` will be a (nested) tuple having the same
                nest structure as the cell output.

              - :attr:`final_state`: The final state of the RNN, which is a
                Tensor of shape ``[batch_size] + cell.state_size`` or
                a (nested) tuple of Tensors if ``cell.state_size`` is a
                (nested) tuple.

            - If ``return_cell_output`` is True, returns a triple
              :attr:`(outputs, final_state, cell_outputs)`

              - :attr:`cell_outputs`: The outputs by the RNN cell prior to the
                output layer, having the same structure with :attr:`outputs`
                except for the ``output_dim``.

            - If ``return_output_size`` is `True`, returns a tuple
              :attr:`(outputs, final_state, output_size)`

              - :attr:`output_size`: A (possibly nested tuple of) int
                representing the size of :attr:`outputs`. If a single int or
                an int array, then ``outputs`` has shape
                ``[batch/time, time/batch] + output_size``. If
                a (nested) tuple, then ``output_size`` has the same
                structure as with ``outputs``.

            - If both ``return_cell_output`` and ``return_output_size`` are
              `True`, returns
              :attr:`(outputs, final_state, cell_outputs, output_size)`.
        """

        cell_outputs, state = dynamic_rnn(
            cell=self._cell,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state=initial_state,
            time_major=time_major)

        outputs, output_size = _forward_output_layers(
            inputs=cell_outputs,
            output_layer=self._output_layer,
            time_major=time_major,
            sequence_length=sequence_length)

        rets = (outputs, state)
        if return_cell_output:
            rets += (cell_outputs,)  # type: ignore
        if return_output_size:
            rets += (output_size,)  # type: ignore
        return rets

    @property
    def cell(self) -> RNNCellBase[State]:
        r"""The RNN cell.
        """
        return self._cell

    @property
    def state_size(self) -> int:
        r"""The state size of encoder cell.
        Same as :attr:`encoder.cell.state_size`.
        """
        if isinstance(self._cell, LSTMCell):
            return 2 * self._cell.hidden_size  # type: ignore
        else:
            return self._cell.hidden_size

    @property
    def output_layer(self) -> Optional[nn.Module]:
        r"""The output layer.
        """
        return self._output_layer

    @property
    def output_size(self) -> int:
        r"""The feature size of :meth:`forward` output :attr:`outputs`.
        If output layer does not exist, the feature size is equal to
        :attr:`encoder.cell.hidden_size`, otherwise the feature size
        is equal to last dimension value of output layer output size.
        """
        # TODO: We will change the implementation to
        # something that does not require a forward pass.

        dim = self._cell.hidden_size
        if self._output_layer is not None:
            dummy_tensor = torch.Tensor(dim)
            dim = self._output_layer(dummy_tensor).size(-1)
        return dim


class BidirectionalRNNEncoder(RNNEncoderBase):
    r"""Bidirectional forward-backward RNN encoder.

    Args:
        cell_fw (RNNCell, optional): The forward RNN cell. If not given,
            a cell is created as specified in ``hparams["rnn_cell_fw"]``.
        cell_bw (RNNCell, optional): The backward RNN cell. If not given,
            a cell is created as specified in ``hparams["rnn_cell_bw"]``.
        output_layer_fw (optional): An instance of
            :torch_nn:`Module`. Apply to the forward
            RNN cell output of each step. If `None` (default), the output
            layer is created as specified in ``hparams["output_layer_fw"]``.
        output_layer_bw (optional): An instance of
            :torch_nn:`Module`. Apply to the backward
            RNN cell output of each step. If `None` (default), the output
            layer is created as specified in ``hparams["output_layer_bw"]``.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    See :meth:`forward` for the inputs and outputs of the encoder.

    Example:

        .. code-block:: python

            # Use with embedder
            embedder = WordEmbedder(vocab_size, hparams=emb_hparams)
            encoder = BidirectionalRNNEncoder(hparams=enc_hparams)

            outputs, final_state = encoder(
                inputs=embedder(data_batch['text_ids']),
                sequence_length=data_batch['length'])
            # outputs == (outputs_fw, outputs_bw)
            # final_state == (final_state_fw, final_state_bw)

    .. document private functions
    """

    def __init__(self,
                 input_size: int,
                 cell_fw: Optional[RNNCellBase[State]] = None,
                 cell_bw: Optional[RNNCellBase[State]] = None,
                 output_layer_fw: Optional[nn.Module] = None,
                 output_layer_bw: Optional[nn.Module] = None,
                 hparams=None):
        super().__init__(hparams=hparams)

        # Make RNN cells
        if cell_fw is not None:
            self._cell_fw = cell_fw
        else:
            self._cell_fw = layers.get_rnn_cell(input_size,
                                                self._hparams.rnn_cell_fw)

        if cell_bw is not None:
            self._cell_bw = cell_bw
        elif self._hparams.rnn_cell_share_config:
            self._cell_bw = layers.get_rnn_cell(input_size,
                                                self._hparams.rnn_cell_fw)
        else:
            self._cell_bw = layers.get_rnn_cell(input_size,
                                                self._hparams.rnn_cell_bw)

        # Make output layers

        self.__output_layer_fw: Optional[nn.Module]
        if output_layer_fw is not None:
            self._output_layer_fw = output_layer_fw
            self._output_layer_hparams_fw = None
        else:
            self._output_layer_fw = _build_dense_output_layer(  # type: ignore
                self._cell_fw.hidden_size, self._hparams.output_layer_fw)
            self._output_layer_hparams_fw = self._hparams.output_layer_fw

        self.__output_layer_bw: Optional[nn.Module]
        if output_layer_bw is not None:
            self._output_layer_bw = output_layer_bw
            self._output_layer_hparams_bw = None
        elif self._hparams.output_layer_share_config:
            self._output_layer_bw = _build_dense_output_layer(  # type: ignore
                self._cell_bw.hidden_size, self._hparams.output_layer_fw)
            self._output_layer_hparams_bw = self._hparams.output_layer_fw
        else:
            self._output_layer_bw = _build_dense_output_layer(  # type: ignore
                self._cell_bw.hidden_size, self._hparams.output_layer_bw)
            self._output_layer_hparams_bw = self._hparams.output_layer_bw

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "rnn_cell_fw": default_rnn_cell_hparams(),
                "rnn_cell_bw": default_rnn_cell_hparams(),
                "rnn_cell_share_config": True,
                "output_layer_fw": {
                    "num_layers": 0,
                    "layer_size": 128,
                    "activation": "identity",
                    "final_layer_activation": None,
                    "other_dense_kwargs": None,
                    "dropout_layer_ids": [],
                    "dropout_rate": 0.5,
                    "variational_dropout": False
                },
                "output_layer_bw": {
                    # Same hyperparams and default values as "output_layer_fw"
                    # ...
                },
                "output_layer_share_config": True,
                "name": "bidirectional_rnn_encoder"
            }

        Here:

        `"rnn_cell_fw"`: dict
            Hyperparameters of the forward RNN cell.
            Ignored if :attr:`cell_fw` is given to the encoder constructor.

            The default value is defined in
            :func:`~texar.torch.core.default_rnn_cell_hparams`.

        `"rnn_cell_bw"`: dict
            Hyperparameters of the backward RNN cell.
            Ignored if :attr:`cell_bw` is given to the encoder constructor,
            or if `"rnn_cell_share_config"` is `True`.

            The default value is defined in
            :meth:`~texar.torch.core.default_rnn_cell_hparams`.

        `"rnn_cell_share_config"`: bool
            Whether share hyperparameters of the backward cell with the
            forward cell. Note that the cell parameters (variables) are not
            shared.

        `"output_layer_fw"`: dict
            Hyperparameters of the forward output layer. Ignored if
            ``output_layer_fw`` is given to the constructor.
            See the ``"output_layer"`` field of
            :meth:`~texar.torch.modules.UnidirectionalRNNEncoder` for details.

        `"output_layer_bw"`: dict
            Hyperparameters of the backward output layer. Ignored if
            :attr:`output_layer_bw` is given to the constructor. Have the
            same structure and defaults with :attr:`"output_layer_fw"`.

            Ignored if ``output_layer_share_config`` is `True`.

        `"output_layer_share_config"`: bool
            Whether share hyperparameters of the backward output layer
            with the forward output layer. Note that the layer parameters
            (variables) are not shared.

        `"name"`: str
            Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams.update({
            "rnn_cell_fw": layers.default_rnn_cell_hparams(),
            "rnn_cell_bw": layers.default_rnn_cell_hparams(),
            "rnn_cell_share_config": True,
            "output_layer_fw": _default_output_layer_hparams(),
            "output_layer_bw": _default_output_layer_hparams(),
            "output_layer_share_config": True,
            "name": "bidirectional_rnn_encoder"
        })
        return hparams

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                sequence_length: Optional[Union[torch.LongTensor,
                                                List[int]]] = None,
                initial_state_fw: Optional[State] = None,
                initial_state_bw: Optional[State] = None,
                time_major: bool = False,
                return_cell_output: bool = False,
                return_output_size: bool = False):
        r"""Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape ``[batch_size, max_time, dim]``.
                The first two dimensions
                ``batch_size`` and ``max_time`` may be exchanged if
                ``time_major`` is `True`.
            sequence_length (optional): A 1D :tensor:`LongTensor` of shape
                ``[batch_size]``.
                Sequence lengths of the batch inputs. Used to copy-through
                state and zero-out outputs when past a batch element's sequence
                length.
            initial_state_fw: (optional): Initial state of the forward RNN.
            initial_state_bw: (optional): Initial state of the backward RNN.
            time_major (bool): The shape format of the :attr:`inputs` and
                :attr:`outputs` Tensors. If `True`, these tensors are of shape
                ``[max_time, batch_size, depth]``. If `False` (default),
                these tensors are of shape ``[batch_size, max_time, depth]``.
            return_cell_output (bool): Whether to return the output of the RNN
                cell. This is the results prior to the output layer.
            return_output_size (bool): Whether to return the output size of the
                RNN cell. This is the results after the output layer.

        Returns:
            - By default (both ``return_cell_output`` and ``return_output_size``
              are `False`), returns a pair :attr:`(outputs, final_state)`

              - :attr:`outputs`: A tuple ``(outputs_fw, outputs_bw)``
                containing the forward and the backward RNN outputs, each of
                which is of shape ``[batch_size, max_time, output_dim]``
                if ``time_major`` is `False`, or
                ``[max_time, batch_size, output_dim]`` if ``time_major``
                is `True`.
                If RNN cell output is a (nested) tuple of Tensors, then
                ``outputs_fw`` and ``outputs_bw`` will be a (nested) tuple
                having the same structure as the cell output.

              - :attr:`final_state`: A tuple
                ``(final_state_fw, final_state_bw)`` containing the final
                states of the forward and backward RNNs, each of which is a
                Tensor of shape ``[batch_size] + cell.state_size``, or a
                (nested) tuple of Tensors if ``cell.state_size`` is a (nested)
                tuple.

            - If ``return_cell_output`` is `True`, returns a triple
              :attr:`(outputs, final_state, cell_outputs)` where

              - :attr:`cell_outputs`: A tuple
                ``(cell_outputs_fw, cell_outputs_bw)`` containing the outputs
                by the forward and backward RNN cells prior to the output
                layers, having the same structure with :attr:`outputs` except
                for the ``output_dim``.

            - If ``return_output_size`` is `True`, returns a tuple
              :attr:`(outputs, final_state, output_size)` where

              - :attr:`output_size`: A tuple
                ``(output_size_fw, output_size_bw)`` containing the size of
                ``outputs_fw`` and ``outputs_bw``, respectively.
                Take ``*_fw`` for example, ``output_size_fw`` is a (possibly
                nested tuple of) int. If a single int or an int array, then
                ``outputs_fw`` has shape
                ``[batch/time, time/batch] + output_size_fw``. If a (nested)
                tuple, then ``output_size_fw`` has the same structure as
                ``outputs_fw``. The same applies to ``output_size_bw``.

            - If both ``return_cell_output`` and ``return_output_size`` are
              `True`, returns
              :attr:`(outputs, final_state, cell_outputs, output_size)`.
        """

        cell_outputs, states = bidirectional_dynamic_rnn(
            cell_fw=self._cell_fw,
            cell_bw=self._cell_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            time_major=time_major)

        outputs_fw, output_size_fw = _forward_output_layers(
            inputs=cell_outputs[0],
            output_layer=self._output_layer_fw,
            time_major=time_major,
            sequence_length=sequence_length)

        outputs_bw, output_size_bw = _forward_output_layers(
            inputs=cell_outputs[1],
            output_layer=self._output_layer_bw,
            time_major=time_major,
            sequence_length=sequence_length)

        outputs = (outputs_fw, outputs_bw)
        output_size = (output_size_fw, output_size_bw)

        returns = (outputs, states)
        if return_cell_output:
            returns += (cell_outputs,)  # type: ignore
        if return_output_size:
            returns += (output_size,)  # type: ignore
        return returns

    @property
    def cell_fw(self) -> RNNCellBase[State]:
        r"""The forward RNN cell.
        """
        return self._cell_fw

    @property
    def cell_bw(self) -> RNNCellBase[State]:
        r"""The backward RNN cell.
        """
        return self._cell_bw

    @property
    def state_size_fw(self) -> int:
        r"""The state size of the forward encoder cell.
        Same as :attr:`encoder.cell_fw.state_size`.
        """
        if isinstance(self._cell_fw, LSTMCell):
            return 2 * self._cell_fw.hidden_size  # type: ignore
        else:
            return self._cell_fw.hidden_size

    @property
    def state_size_bw(self) -> int:
        r"""The state size of the backward encoder cell.
        Same as :attr:`encoder.cell_bw.state_size`.
        """
        if isinstance(self._cell_bw, LSTMCell):
            return 2 * self._cell_bw.hidden_size  # type: ignore
        else:
            return self._cell_bw.hidden_size

    @property
    def output_layer_fw(self) -> Optional[nn.Module]:
        r"""The output layer of the forward RNN.
        """
        return self._output_layer_fw

    @property
    def output_layer_bw(self) -> Optional[nn.Module]:
        r"""The output layer of the backward RNN.
        """
        return self._output_layer_bw

    @property
    def output_size(self) -> Tuple[int, int]:
        r"""The feature sizes of :meth:`forward` outputs
        :attr:`output_size_fw` and :attr:`output_size_bw`.
        Each feature size is equal to last dimension
        value of corresponding result size.
        """
        # TODO: We will change the implementation to
        # something that does not require a forward pass.
        dim_bw = self._cell_bw.hidden_size
        dim_fw = self._cell_fw.hidden_size
        if self._output_layer_bw is not None:
            dummy_tensor_bw = torch.Tensor(dim_bw)
            output_bw = self._output_layer_bw(dummy_tensor_bw).size()[-1]
        else:
            output_bw = dim_bw
        if self._output_layer_fw is not None:
            dummy_tensor_fw = torch.Tensor(dim_fw)
            output_fw = self._output_layer_fw(dummy_tensor_fw).size()[-1]
        else:
            output_fw = dim_fw
        return (output_fw, output_bw)
