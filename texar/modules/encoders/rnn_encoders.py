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

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from texar.core.cell_wrappers import LSTMCell
from texar import HParams
from texar import ModuleBase
from texar.modules.encoders import EncoderBase
from texar.modules.networks.conv_networks import _to_list
from texar.core import layers, RNNCellBase
from texar.utils.shapes import mask_sequences
from texar.utils.rnn import dynamic_rnn, bidirectional_dynamic_rnn


# pylint: disable=too-many-arguments, too-many-locals, invalid-name, no-member

__all__ = [
    "_forward_single_output_layer",
    "RNNEncoderBase",
    "UnidirectionalRNNEncoder",
    "BidirectionalRNNEncoder"
]


def _default_output_layer_hparams():
    return {
        "num_layers": 0,
        "layer_size": 128,
        "activation": "identity",
        "final_layer_activation": None,
        "other_dense_kwargs": None,
        "dropout_layer_ids": [],
        "dropout_rate": 0.5,
        "variational_dropout": False,
        "@no_typecheck": ["activation", "final_layer_activation",
                          "layer_size", "dropout_layer_ids"]
    }


def _build_dense_output_layer(cell_output_size, hparams):
    nlayers = hparams.num_layers

    if nlayers <= 0:
        return None

    layer_size = _to_list(
        hparams.layer_size, 'output_layer.layer_size', nlayers)

    other_kwargs = hparams.other_dense_kwargs or {}
    if isinstance(other_kwargs, HParams):
        other_kwargs = other_kwargs.todict()
    if not isinstance(other_kwargs, dict):
        raise ValueError(
            "hparams 'output_layer.other_dense_kwargs' must be a dict.")

    dense_layers = []
    for i in range(nlayers):
        if i == 0:
            kwargs_i = {"in_features": cell_output_size,
                        "out_features": layer_size[i]}
        else:
            kwargs_i = {"in_features": layer_size[i-1],
                        "out_features": layer_size[i]}

        kwargs_i.update(other_kwargs)

        layer_hparams = {"type": nn.Linear, "kwargs": kwargs_i}
        dense_layer = layers.get_layer(hparams=layer_hparams)
        # dense_layers.append(layers.get_layer(hparams=layer_hparams))

        if i == nlayers - 1:
            activation = hparams.final_layer_activation
        else:
            activation = hparams.activation

        if activation is not None:
            layer_hparams = {"type": activation, "kwargs": {}}
            activation_layer = layers.get_layer(hparams=layer_hparams)
            #  dense_layers.append(layers.get_layer(hparams=layer_hparams))
        else:
            activation_layer = None

        dense_layers.append((dense_layer, activation_layer))

    if len(dense_layers) == 1:
        dense_layers = dense_layers[0]

    return dense_layers


def _forward_single_output_layer(inputs, output_layer):
    """Forwards the input through a single output layer.

    Args:
        inputs: A Tensor of shape `[batch_size, max_time] + input_size` if
            :attr:`time_major=False`, or shape
            `[max_time, batch_size] + input_size` if :attr:`time_major=True`.
        input_size: An `int` or 1D `int` array.
        output_layer: A tuple of (Linear Layer, Activation).
    """
    layer, activation = output_layer
    # Feed to the layer
    output = layer(inputs)
    if activation is not None:
        output = activation(output)
    output_size = output.shape[-1]
    return output, output_size


def _apply_dropout(inputs, time_major, hparams, training):
    """Applies dropout to the inputs.

    :attr:`inputs` is a Tensor of shape `[batch_size, max_time, dim]`
    if :attr:`time_major=False`, or shape `[max_time, batch_size, dim]`
    if :attr:`time_major=True`.
    """
    noise_shape = None
    # TODO: Implement variational dropout layer
    if hparams.variational_dropout:
        if time_major:
            raise NotImplementedError
        else:
            raise NotImplementedError

    return F.dropout(inputs, p=hparams.dropout_rate, training=training)


def _forward_output_layers(inputs, input_size, output_layer, time_major,
                           hparams, mode, sequence_length=None):
    """Forwards inputs through the output layers.

    Args:
        inputs: A Tensor of shape `[batch_size, max_time] + input_size` if
            :attr:`time_major=False`, or shape
            `[max_time, batch_size] + input_size` if :attr:`time_major=True`.

    Returns:
        A pair :attr:`(outputs, outputs_size), where

        - :attr:`outputs`: A Tensor of shape \
          `[batch_size, max_time] + outputs_size`.

        - :attr:`outputs_size`: An `int` or 1D `int` array representing the \
          output size.
    """
    if output_layer is None:
        return inputs, input_size

    if hparams is None:
        # output_layer was passed in from the constructor
        if isinstance(output_layer, list):
            raise ValueError('output_layer must not be a list.')
        output, output_size = _forward_single_output_layer(inputs, output_layer)
    else:
        # output_layer was built based on hparams
        output_layer = _to_list(output_layer)

        dropout_layer_ids = _to_list(hparams.dropout_layer_ids)
        if len(dropout_layer_ids) > 0:
            training = mode

        output = inputs
        output_size = input_size
        for i, layer in enumerate(output_layer):
            if i in dropout_layer_ids:
                output = _apply_dropout(output, time_major, hparams, training)
            output, output_size = _forward_single_output_layer(output, layer)

        if len(output_layer) in dropout_layer_ids:
            output = _apply_dropout(output, hparams, training)

    if sequence_length is not None:
        output = mask_sequences(output, sequence_length, time_major=time_major)

    return output, output_size


def _apply_rnn_encoder_output_layer(output_layer, time_major, hparams, mode,
                                    cell_outputs, cell_output_size):
    map_func = functools.partial(
        _forward_output_layers,
        output_layer=output_layer,
        time_major=time_major,
        hparams=hparams,
        mode=mode)

    output, output_size = map_func(inputs=cell_outputs,
                                   input_size=cell_output_size)

    return output, output_size


class RNNEncoderBase(EncoderBase):
    """Base class for all RNN encoder classes to inherit.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self, hparams: Optional[HParams] = None):
        super().__init__(hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "rnn_encoder"
            }
        """
        return {
            'name': 'rnn_encoder'
        }

    def forward(self, inputs, *args, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: Inputs to the encoder.
            *args: Other arguments.
            **kwargs: Keyword arguments.

        Returns:
            Encoding results.
        """
        raise NotImplementedError


class UnidirectionalRNNEncoder(RNNEncoderBase):
    """One directional RNN encoder.

    Args:
        cell: (RNNCell, optional) If not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in
            TRAIN mode). If `None`, :func:`~texar.global_mode` is used.
            Ignored if :attr:`cell` is given.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`. Applies to the RNN cell
            output of each step. If `None` (default), the output layer is
            created as specified in :attr:`hparams["output_layer"]`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`_build` for the inputs and outputs of the encoder.

    Example:

        .. code-block:: python

            # Use with embedder
            embedder = WordEmbedder(vocab_size, hparams=emb_hparams)
            encoder = UnidirectionalRNNEncoder(hparams=enc_hparams)

            outputs, final_state = encoder(
                inputs=embedder(data_batch['text_ids']),
                sequence_length=data_batch['length'])

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 input_size: int,
                 cell: Optional[RNNCellBase] = None,
                 output_layer: Optional[nn.Module] = None,
                 hparams: Optional[HParams] = None):
        super().__init__(hparams)

        # Make RNN cell
        if cell is not None:
            self._cell = cell
        else:
            self._cell = layers.get_rnn_cell(input_size,
                                             self._hparams.rnn_cell)

        # Make output layer
        if output_layer is not None:
            self._output_layer = output_layer
            self._output_layer_hparams = None
        else:
            self._output_layer = _build_dense_output_layer(
                self._cell.hidden_size, self._hparams.output_layer)
            self._output_layer_hparams = self._hparams.output_layer

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

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

        "rnn_cell" : dict
            A dictionary of RNN cell hyperparameters. Ignored if
            :attr:`cell` is given to the encoder constructor.

            The default value is defined in
            :func:`~texar.core.default_rnn_cell_hparams`.

        "output_layer" : dict
            Output layer hyperparameters. Ignored if :attr:`output_layer`
            is given to the encoder constructor. Includes:

            "num_layers" : int
                The number of output (dense) layers. Set to 0 to avoid any
                output layers applied to the cell outputs..

            "layer_size" : int or list
                The size of each of the output (dense) layers.

                If an `int`, each output layer will have the same size. If
                a list, the length must equal to :attr:`num_layers`.

            "activation" : str or callable or None
                Activation function for each of the output (dense)
                layer except for the final layer. This can be
                a function, or its string name or module path.
                If function name is given, the function must be from
                module :tf_main:`tf.nn <nn>` or :tf_main:`tf < >`.
                For example

                .. code-block:: python

                    "activation": "relu" # function name
                    "activation": "my_module.my_activation_fn" # module path
                    "activation": my_module.my_activation_fn # function

                Default is `None` which maintains a linear activation.

            "final_layer_activation" : str or callable or None
                The activation function for the final output layer.

            "other_dense_kwargs" : dict or None
                Other keyword arguments to construct each of the output
                dense layers, e.g., `use_bias`. See
                :tf_main:`Dense <layers/Dense>` for the keyword arguments.

            "dropout_layer_ids" : int or list
                The indexes of layers (starting from `0`) whose inputs
                are applied with dropout. The index = :attr:`num_layers`
                means dropout applies to the final layer output. E.g.,

                .. code-block:: python

                    {
                        "num_layers": 2,
                        "dropout_layer_ids": [0, 2]
                    }

                will leads to a series of layers as
                `-dropout-layer0-layer1-dropout-`.

                The dropout mode (training or not) is controlled
                by the :attr:`mode` argument of :meth:`_build`.

            "dropout_rate" : float
                The dropout rate, between 0 and 1. E.g.,
                `"dropout_rate": 0.1` would drop out 10% of elements.

            "variational_dropout": bool
                Whether the dropout mask is the same across all time steps.

        "name" : str
            Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams.update({
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "output_layer": _default_output_layer_hparams(),
            "name": "unidirectional_rnn_encoder"
        })
        return hparams

    def forward(self,
                inputs,
                sequence_length=None,
                initial_state=None,
                time_major=False,
                return_cell_output=False,
                return_output_size=False):
        """Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`.
                The first two dimensions
                :attr:`batch_size` and :attr:`max_time` are exchanged if
                :attr:`time_major=True` is specified.
            sequence_length (optional): A 1D int tensor of shape `[batch_size]`.
                Sequence lengths
                of the batch inputs. Used to copy-through state and zero-out
                outputs when past a batch element's sequence length.
            initial_state (optional): Initial state of the RNN.
            time_major (bool): The shape format of the :attr:`inputs` and
                :attr:`outputs` Tensors. If `True`, these tensors are of shape
                `[max_time, batch_size, depth]`. If `False` (default),
                these tensors are of shape `[batch_size, max_time, depth]`.
            return_cell_output (bool): Whether to return the output of the RNN
                cell. This is the results prior to the output layer.
            return_output_size (bool): Whether to return the size of the
                output (i.e., the results after output layers).

        Returns:
            - By default (both `return_cell_output` and \
            `return_output_size` are False), returns a pair \
            :attr:`(outputs, final_state)`

                - :attr:`outputs`: The RNN output tensor by the output layer \
                (if exists) or the RNN cell (otherwise). The tensor is of \
                shape `[batch_size, max_time, output_size]` if \
                `time_major` is False, or \
                `[max_time, batch_size, output_size]` if \
                `time_major` is True. \
                If RNN cell output is a (nested) tuple of Tensors, then the \
                :attr:`outputs` will be a (nested) tuple having the same \
                nest structure as the cell output.

                - :attr:`final_state`: The final state of the RNN, which is a \
                Tensor of shape `[batch_size] + cell.state_size` or \
                a (nested) tuple of Tensors if `cell.state_size` is a (nested)\
                tuple.

            - If `return_cell_output` is True, returns a triple \
            :attr:`(outputs, final_state, cell_outputs)`

                - :attr:`cell_outputs`: The outputs by the RNN cell prior to \
                the \
                output layer, having the same structure with :attr:`outputs` \
                except for the `output_dim`.

            - If `return_output_size` is `True`, returns a tuple \
            :attr:`(outputs, final_state, output_size)`

                - :attr:`output_size`: A (possibly nested tuple of) int \
                representing the size of :attr:`outputs`. If a single int or \
                an int array, then `outputs` has shape \
                `[batch/time, time/batch] + output_size`. If \
                a (nested) tuple, then `output_size` has the same \
                structure as with `outputs`.

            - If both `return_cell_output` and \
            `return_output_size` are True, returns \
            :attr:`(outputs, final_state, cell_outputs, output_size)`.
        """

        cell_outputs, state = dynamic_rnn(
            cell=self._cell,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state=initial_state,
            time_major=time_major)

        outputs, output_size = _apply_rnn_encoder_output_layer(
            self._output_layer, time_major, self._output_layer_hparams,
            self.training, cell_outputs, self._cell.hidden_size)

        rets = (outputs, state)
        if return_cell_output:
            rets += (cell_outputs, )
        if return_output_size:
            rets += (output_size, )
        return rets

    @property
    def cell(self):
        """The RNN cell.
        """
        return self._cell

    @property
    def state_size(self):
        """The state size of encoder cell.
        Same as :attr:`encoder.cell.state_size`.
        """
        if isinstance(self._cell, LSTMCell):
            return 2 * self._cell.hidden_size
        else:
            return self._cell.hidden_size

    @property
    def output_layer(self):
        """The output layer.
        """
        return self._output_layer


class BidirectionalRNNEncoder(RNNEncoderBase):
    """Bidirectional forward-backward RNN encoder.

    Args:
        cell_fw (RNNCell, optional): The forward RNN cell. If not given,
            a cell is created as specified in :attr:`hparams["rnn_cell_fw"]`.
        cell_bw (RNNCell, optional): The backward RNN cell. If not given,
            a cell is created as specified in :attr:`hparams["rnn_cell_bw"]`.
        cell_dropout_mode (optional): A tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cells (e.g., activates dropout in
            TRAIN mode). If `None`, :func:`~texar.global_mode()` is
            used. Ignored if respective cell is given.
        output_layer_fw (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`. Apply to the forward
            RNN cell output of each step. If `None` (default), the output
            layer is created as specified in :attr:`hparams["output_layer_fw"]`.
        output_layer_bw (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`. Apply to the backward
            RNN cell output of each step. If `None` (default), the output
            layer is created as specified in :attr:`hparams["output_layer_bw"]`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`_build` for the inputs and outputs of the encoder.

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
    .. automethod:: _build
    """

    def __init__(self,
                 input_size,
                 cell_fw=None,
                 cell_bw=None,
                 output_layer_fw=None,
                 output_layer_bw=None,
                 hparams=None):
        super().__init__(hparams)

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
        if output_layer_fw is not None:
            self._output_layer_fw = output_layer_fw
            self._output_layer_hparams_fw = None
        else:
            self._output_layer_fw = _build_dense_output_layer(
                self._cell_fw.hidden_size, self._hparams.output_layer_fw)
            self._output_layer_hparams_fw = self._hparams.output_layer_fw

        if output_layer_bw is not None:
            self._output_layer_bw = output_layer_bw
            self._output_layer_hparams_bw = None
        elif self._hparams.output_layer_share_config:
            self._output_layer_bw = _build_dense_output_layer(
                self._cell_bw.hidden_size, self._hparams.output_layer_fw)
            self._output_layer_hparams_bw = self._hparams.output_layer_fw
        else:
            self._output_layer_bw = _build_dense_output_layer(
                self._cell_bw.hidden_size, self._hparams.output_layer_bw)
            self._output_layer_hparams_bw = self._hparams.output_layer_bw

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

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

        "rnn_cell_fw" : dict
            Hyperparameters of the forward RNN cell.
            Ignored if :attr:`cell_fw` is given to the encoder constructor.

            The default value is defined in
            :func:`~texar.core.default_rnn_cell_hparams`.

        "rnn_cell_bw" : dict
            Hyperparameters of the backward RNN cell.
            Ignored if :attr:`cell_bw` is given to the encoder constructor
            , or if :attr:`"rnn_cell_share_config"` is `True`.

            The default value is defined in
            :meth:`~texar.core.default_rnn_cell_hparams`.

        "rnn_cell_share_config" : bool
            Whether share hyperparameters of the backward cell with the
            forward cell. Note that the cell parameters (variables) are not
            shared.

        "output_layer_fw" : dict
            Hyperparameters of the forward output layer. Ignored if
            :attr:`output_layer_fw` is given to the constructor.
            See the "output_layer" field of
            :meth:`~texar.modules.UnidirectionalRNNEncoder.default_hparams` for
            details.

        "output_layer_bw" : dict
            Hyperparameters of the backward output layer. Ignored if
            :attr:`output_layer_bw` is given to the constructor. Have the
            same structure and defaults with :attr:`"output_layer_fw"`.

            Ignored if :attr:`"output_layer_share_config"` is True.

        "output_layer_share_config" : bool
            Whether share hyperparameters of the backward output layer
            with the forward output layer. Note that the layer parameters
            (variables) are not shared.

        "name" : str
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

    def forward(self,
                inputs,
                sequence_length=None,
                initial_state_fw=None,
                initial_state_bw=None,
                time_major=False,
                return_cell_output=False,
                return_output_size=False):
        """Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`.
                The first two dimensions
                `batch_size` and `max_time` may be exchanged if
                `time_major=True` is specified.
            sequence_length (optional): A 1D int tensor of shape `[batch_size]`.
                Sequence lengths
                of the batch inputs. Used to copy-through state and zero-out
                outputs when past a batch element's sequence length.
            initial_state (optional): Initial state of the RNN.
            time_major (bool): The shape format of the :attr:`inputs` and
                :attr:`outputs` Tensors. If `True`, these tensors are of shape
                `[max_time, batch_size, depth]`. If `False` (default),
                these tensors are of shape `[batch_size, max_time, depth]`.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. Controls output layer dropout
                if the output layer is specified with :attr:`hparams`.
                If `None` (default), :func:`texar.global_mode()`
                is used.
            return_cell_output (bool): Whether to return the output of the RNN
                cell. This is the results prior to the output layer.
            **kwargs: Optional keyword arguments of
                :tf_main:`tf.nn.dynamic_rnn <nn/dynamic_rnn>`,
                such as `swap_memory`, `dtype`, `parallel_iterations`, etc.

        Returns:
            - By default (both `return_cell_output` and `return_output_size` \
            are False), returns a pair :attr:`(outputs, final_state)`

                - :attr:`outputs`: A tuple `(outputs_fw, outputs_bw)` \
                containing \
                the forward and the backward RNN outputs, each of which is of \
                shape `[batch_size, max_time, output_dim]` if \
                `time_major` is False, or \
                `[max_time, batch_size, output_dim]` if \
                `time_major` is True. \
                If RNN cell output is a (nested) tuple of Tensors, then \
                `outputs_fw` and `outputs_bw` will be a (nested) tuple having \
                the same structure as the cell output.

                - :attr:`final_state`: A tuple \
                `(final_state_fw, final_state_bw)` \
                containing the final states of the forward and backward \
                RNNs, each of which is a \
                Tensor of shape `[batch_size] + cell.state_size`, or \
                a (nested) tuple of Tensors if `cell.state_size` is a (nested)\
                tuple.

            - If `return_cell_output` is True, returns a triple \
            :attr:`(outputs, final_state, cell_outputs)` where

                - :attr:`cell_outputs`: A tuple \
                `(cell_outputs_fw, cell_outputs_bw)` containting the outputs \
                by the forward and backward RNN cells prior to the \
                output layers, having the same structure with :attr:`outputs` \
                except for the `output_dim`.

            - If `return_output_size` is True, returns a tuple \
            :attr:`(outputs, final_state, output_size)` where

                - :attr:`output_size`: A tupple \
                `(output_size_fw, output_size_bw)` containing the size of \
                `outputs_fw` and `outputs_bw`, respectively. \
                Take `*_fw` for example, \
                `output_size_fw` is a (possibly nested tuple of) int. \
                If a single int or an int array, then `outputs_fw` has shape \
                `[batch/time, time/batch] + output_size_fw`. If \
                a (nested) tuple, then `output_size_fw` has the same \
                structure as with `outputs_fw`. The same applies to  \
                `output_size_bw`.

            - If both `return_cell_output` and \
            `return_output_size` are True, returns \
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

        outputs_fw, output_size_fw = _apply_rnn_encoder_output_layer(
            self._output_layer_fw, time_major, self._output_layer_hparams_fw,
            self.training, cell_outputs[0], self._cell_fw.hidden_size)

        outputs_bw, output_size_bw = _apply_rnn_encoder_output_layer(
            self._output_layer_bw, time_major, self._output_layer_hparams_bw,
            self.training, cell_outputs[1], self._cell_bw.hidden_size)

        outputs = (outputs_fw, outputs_bw)
        output_size = (output_size_fw, output_size_bw)

        returns = (outputs, states)
        if return_cell_output:
            returns += (cell_outputs, )
        if return_output_size:
            returns += (output_size,)
        return returns

    @property
    def cell_fw(self):
        """The forward RNN cell.
        """
        return self._cell_fw

    @property
    def cell_bw(self):
        """The backward RNN cell.
        """
        return self._cell_bw

    @property
    def state_size_fw(self):
        """The state size of the forward encoder cell.
        Same as :attr:`encoder.cell_fw.state_size`.
        """
        if isinstance(self._cell_fw, LSTMCell):
            return 2 * self._cell_fw.hidden_size
        else:
            return self._cell_fw.hidden_size

    @property
    def state_size_bw(self):
        """The state size of the backward encoder cell.
        Same as :attr:`encoder.cell_bw.state_size`.
        """
        if isinstance(self._cell_bw, LSTMCell):
            return 2 * self._cell_bw.hidden_size
        else:
            return self._cell_bw.hidden_size

    @property
    def output_layer_fw(self):
        """The output layer of the forward RNN.
        """
        return self._output_layer_fw

    @property
    def output_layer_bw(self):
        """The output layer of the backward RNN.
        """
        return self._output_layer_bw
