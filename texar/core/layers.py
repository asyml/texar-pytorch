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
Various neural network layers
"""

import torch
from torch import nn

import texar.core.cell_wrappers as wrappers
from texar.hyperparams import HParams
from texar.utils import utils

# pylint: disable=too-many-branches

__all__ = [
    "default_rnn_cell_hparams",
    "get_rnn_cell",
    "identity",
]


def default_rnn_cell_hparams():
    r"""Returns a `dict` of RNN cell hyperparameters and their default values.

    .. role:: python(code)
       :language: python

    .. code-block:: python

        {
            "type": "LSTMCell",
            "input_size": 256,
            "kwargs": {
                "hidden_size": 256
            },
            "num_layers": 1,
            "dropout": {
                "input_keep_prob": 1.0,
                "output_keep_prob": 1.0,
                "state_keep_prob": 1.0,
                "variational_recurrent": False,
            },
            "residual": False,
            "highway": False,
        }

    Here:

    "type" : str or cell class or cell instance
        The RNN cell type. This can be

        - The string name or full module path of a cell class. If class \
        name is provided, the class must be in module \
        :mod:`torch.nn.modules.rnn`, \
        :mod:`texar.core.cell_wrappers`, or :mod:`texar.custom`.
        - A cell class.
        - An instance of a cell class. This is not valid if \
        "num_layers" > 1.

        For example

        .. code-block:: python

            "type": "LSTMCell"  # class name
            "type": "torch.nn.GRUCell"  # module path
            "type": "my_module.MyCell"  # module path
            "type": torch.nn.GRUCell  # class
            "type": LSTMCell(hidden_size=100)  # cell instance
            "type": MyCell(...)  # cell instance

    "input_size": int
        Size of the input to the cell in the first layer.

    "kwargs" : dict
        Keyword arguments for the constructor of the cell class.
        A cell is created by :python:`cell_class(**kwargs)`, where
        `cell_class` is specified in "type" above.

        Ignored if "type" is a cell instance.

        ..note:: It is unnecessary to specify "input_size" within "kwargs".
            This value will be automatically filled based on layer index.

    "num_layers" : int
        Number of cell layers. Each layer is a cell created as above, with
        the same hyperparameters specified in "kwargs".

    "dropout" : dict
        Dropout applied to the cell in **each** layer. See
        :class:`~texar.core.cell_wrappers.DropoutWrapper` for details of
        the hyperparameters. If all "\*_keep_prob" = 1, no dropout is applied.

        Specifically, if "variational_recurrent" = `True`,
        the same dropout mask is applied across all time steps per batch.

    "residual" : bool
        If `True`, apply residual connection on the inputs and
        outputs of cell in **each** layer except the first layer. Ignored
        if "num_layers" = 1.

    "highway" : bool
        If True, apply highway connection on the inputs and
        outputs of cell in each layer except the first layer. Ignored if
        "num_layers" = 1.
    """
    return {
        "type": "LSTMCell",
        "input_size": 256,
        "kwargs": {
            "hidden_size": 256,
        },
        "num_layers": 1,
        "dropout": {
            "input_keep_prob": 1.0,
            "output_keep_prob": 1.0,
            "state_keep_prob": 1.0,
            "variational_recurrent": False,
        },
        "residual": False,
        "highway": False,
        "@no_typecheck": ["type"]
    }


def get_rnn_cell(hparams=None):
    r"""Creates an RNN cell.

    See :func:`~texar.core.default_rnn_cell_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): Cell hyperparameters. Missing
            hyperparameters are set to default values.

    Returns:
        A cell instance.

    Raises:
        ValueError: If hparams["num_layers"]>1 and hparams["type"] is a class
            instance.
        ValueError: The cell is not an
            :tf_main:`RNNCell <contrib/rnn/RNNCell>` instance.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_rnn_cell_hparams())

    d_hp = hparams['dropout']
    variational_recurrent = d_hp['variational_recurrent']
    input_keep_prob = d_hp['input_keep_prob']
    output_keep_prob = d_hp['output_keep_prob']
    state_keep_prob = d_hp['state_keep_prob']

    cells = []
    cell_kwargs = hparams['kwargs'].todict()
    num_layers = hparams['num_layers']
    for layer_i in range(num_layers):
        # Create the basic cell
        cell_type = hparams["type"]
        if layer_i == 0:
            cell_kwargs['input_size'] = hparams['input_size']
        else:
            cell_kwargs['input_size'] = cell_kwargs['hidden_size']
        if not isinstance(cell_type, str) and not isinstance(cell_type, type):
            if num_layers > 1:
                raise ValueError(
                    "If 'num_layers'>1, then 'type' must be a cell class or "
                    "its name/module path, rather than a cell instance.")
        cell_modules = ['texar.core.cell_wrappers',  # prefer our wrappers
                        'torch.nn.modules.rnn', 'texar.custom']
        cell = utils.check_or_get_instance(cell_type, cell_kwargs, cell_modules)
        if isinstance(cell, nn.RNNCellBase):
            cell = wrappers.wrap_builtin_cell(cell)

        # Optionally add dropout
        if (input_keep_prob < 1.0 or
                output_keep_prob < 1.0 or
                state_keep_prob < 1.0):
            # TODO: Would this result in non-final layer outputs being
            #       dropped twice?
            cell = wrappers.DropoutWrapper(
                cell=cell,
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob,
                state_keep_prob=state_keep_prob,
                variational_recurrent=variational_recurrent)

        # Optionally add residual and highway connections
        if layer_i > 0:
            if hparams['residual']:
                cell = wrappers.ResidualWrapper(cell)
            if hparams['highway']:
                cell = wrappers.HighwayWrapper(cell)

        cells.append(cell)

    if hparams['num_layers'] > 1:
        cell = wrappers.MultiRNNCell(cells)
    else:
        cell = cells[0]

    return cell


def identity(inputs: torch.Tensor):
    """Returns a tensor with the same content as the input tensor.

    Arguments:
        inputs: The input tensor.

    Returns:
        A tensor of the same shape, type, and content.
    """
    return inputs
