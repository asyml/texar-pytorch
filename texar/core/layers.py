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

# pylint: disable=too-many-branches
import functools
from typing import Optional, Callable

import torch
from torch import nn

import texar.core.cell_wrappers as wrappers
from texar.hyperparams import HParams
from texar.utils import utils
from texar.utils.dtypes import is_str

__all__ = [
    'default_rnn_cell_hparams',
    'get_rnn_cell',
    'identity',
    'get_initializer',
    'get_layer'
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

    "kwargs" : dict
        Keyword arguments for the constructor of the cell class.
        A cell is created by :python:`cell_class(**kwargs)`, where
        `cell_class` is specified in "type" above.

        Ignored if "type" is a cell instance.

        ..note:: It is unnecessary to specify "input_size" within "kwargs".
            This value will be automatically filled based on layer index.

        ..note:: Although PyTorch uses "hidden_size" to denote the hidden layer
            size, we follow TensorFlow conventions and use "num_units".

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
        'type': 'LSTMCell',
        'kwargs': {
            'num_units': 256,
        },
        'num_layers': 1,
        'dropout': {
            'input_keep_prob': 1.0,
            'output_keep_prob': 1.0,
            'state_keep_prob': 1.0,
            'variational_recurrent': False,
        },
        'residual': False,
        'highway': False,
        '@no_typecheck': ['type']
    }


def get_rnn_cell(input_size, hparams=None):
    r"""Creates an RNN cell.

    See :func:`~texar.core.default_rnn_cell_hparams` for all
    hyperparameters and default values.

    Args:
        input_size (int): Size of the input to the cell in the first layer.
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
    num_layers = hparams['num_layers']
    cell_kwargs = hparams['kwargs'].todict()
    # rename 'num_units' to 'hidden_size' following PyTorch conventions
    cell_kwargs['hidden_size'] = cell_kwargs['num_units']
    del cell_kwargs['num_units']

    for layer_i in range(num_layers):
        # Create the basic cell
        cell_type = hparams["type"]
        if layer_i == 0:
            cell_kwargs['input_size'] = input_size
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
    r"""Returns a tensor with the same content as the input tensor.

    Arguments:
        inputs: The input tensor.

    Returns:
        A tensor of the same shape, type, and content.
    """
    return inputs


def get_initializer(hparams: Optional[HParams] = None) \
        -> Optional[Callable[[torch.Tensor], None]]:
    r"""Returns an initializer instance.

    .. role:: python(code)
       :language: python

    Args:
        hparams (dict or HParams, optional): Hyperparameters with the structure

            .. code-block:: python

                {
                    "type": "initializer_class_or_function",
                    "kwargs": {
                        #...
                    }
                }

            The "type" field can be a initializer class, its name or module
            path, or class instance. If class name is provided, the class must
            be from one the following modules:
            :tf_main:`tf.initializers <initializers>`,
            :tf_main:`tf.keras.initializers <keras/initializers>`,
            :tf_main:`tf < >`, and :mod:`texar.custom`. The class is created
            by :python:`initializer_class(**kwargs)`. If a class instance
            is given, "kwargs" is ignored and can be omitted.

            Besides, the "type" field can also be an initialization function
            called with :python:`initialization_fn(**kwargs)`. In this case
            "type" can be the function, or its name or module path. If
            function name is provided, the function must be from one of the
            above modules or module `tf.contrib.layers`. If no
            keyword argument is required, "kwargs" can be omitted.

    Returns:
        An initializer instance. `None` if :attr:`hparams` is `None`.
    """
    if hparams is None:
        return None

    kwargs = hparams.get('kwargs', {})
    if isinstance(kwargs, HParams):
        kwargs = kwargs.todict()
    modules = ['torch.nn.init', 'torch', 'texar.custom']
    initializer_fn = utils.get_function(hparams["type"], modules)
    initializer = functools.partial(initializer_fn, **kwargs)

    return initializer

def get_layer(hparams):
    """Makes a layer instance.

    The layer must be an instance of :tf_main:`tf.layers.Layer <layers/Layer>`.

    Args:
        hparams (dict or HParams): Hyperparameters of the layer, with
            structure:

            .. code-block:: python

                {
                    "type": "LayerClass",
                    "kwargs": {
                        # Keyword arguments of the layer class
                        # ...
                    }
                }

            Here:

            "type" : str or layer class or layer instance
                The layer type. This can be

                - The string name or full module path of a layer class. If \
                the class name is provided, the class must be in module \
                :tf_main:`tf.layers <layers>`, :mod:`texar.core`, \
                or :mod:`texar.custom`.
                - A layer class.
                - An instance of a layer class.

                For example

                .. code-block:: python

                    "type": "Conv1D" # class name
                    "type": "texar.core.MaxReducePooling1D" # module path
                    "type": "my_module.MyLayer" # module path
                    "type": tf.layers.Conv2D # class
                    "type": Conv1D(filters=10, kernel_size=2) # cell instance
                    "type": MyLayer(...) # cell instance

            "kwargs" : dict
                A dictionary of keyword arguments for constructor of the
                layer class. Ignored if :attr:`"type"` is a layer instance.

                - Arguments named "activation" can be a callable, \
                or a `str` of \
                the name or module path to the activation function.
                - Arguments named "\*_regularizer" and "\*_initializer" \
                can be a class instance, or a `dict` of \
                hyperparameters of \
                respective regularizers and initializers. See
                - Arguments named "\*_constraint" can be a callable, or a \
                `str` of the name or full path to the constraint function.

    Returns:
        A layer instance. If hparams["type"] is a layer instance, returns it
        directly.

    Raises:
        ValueError: If :attr:`hparams` is `None`.
        ValueError: If the resulting layer is not an instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`.
    """
    if hparams is None:
        raise ValueError("`hparams` must not be `None`.")

    layer_type = hparams["type"]
    if not is_str(layer_type) and not isinstance(layer_type, type):
        layer = layer_type
    else:
        layer_modules = ["tensorflow.layers", "texar.core", "texar.costum"]
        layer_class = utils.check_or_get_class(layer_type, layer_modules)
        if isinstance(hparams, dict):
            default_kwargs = _layer_class_to_default_kwargs_map.get(layer_class,
                                                                    {})
            default_hparams = {"type": layer_type, "kwargs": default_kwargs}
            hparams = HParams(hparams, default_hparams)

        kwargs = {}
        for k, v in hparams.kwargs.items():
            if k.endswith('_regularizer'):
                kwargs[k] = get_regularizer(v)
            elif k.endswith('_initializer'):
                kwargs[k] = get_initializer(v)
            elif k.endswith('activation'):
                kwargs[k] = get_activation_fn(v)
            elif k.endswith('_constraint'):
                kwargs[k] = get_constraint_fn(v)
            else:
                kwargs[k] = v
        layer = utils.get_instance(layer_type, kwargs, layer_modules)

    if not isinstance(layer, tf.layers.Layer):
        raise ValueError("layer must be an instance of `tf.layers.Layer`.")

    return layer
