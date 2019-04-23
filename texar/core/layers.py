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

#import texar.core.cell_wrappers as wrappers
from texar.core import cell_wrappers as wrappers
from texar.hyperparams import HParams
from texar.utils import utils
from texar.utils.dtypes import is_str


__all__ = [
    'default_rnn_cell_hparams',
    'get_rnn_cell',
    'identity',
    'default_regularizer_hparams',
    'get_regularizer',
    'get_initializer',
    'get_activation_fn',
    'get_constraint_fn',
    'get_layer',
    'default_linear_kwargs'
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

def default_regularizer_hparams():
    """Returns the hyperparameters and their default values of a variable
    regularizer:

    .. code-block:: python

        {
            "type": "L1L2",
            "kwargs": {
                "l1": 0.,
                "l2": 0.
            }
        }

    The default value corresponds to :tf_main:`L1L2 <keras/regularizers/L1L2>`
    and, with `(l1=0, l2=0)`, disables regularization.
    """
    return {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0.
        }
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


def default_regularizer_hparams():
    """Returns the hyperparameters and their default values of a variable
    regularizer:

    .. code-block:: python

        {
            "type": "L1L2",
            "kwargs": {
                "l1": 0.,
                "l2": 0.
            }
        }

    The default value corresponds to :tf_main:`L1L2 <keras/regularizers/L1L2>`
    and, with `(l1=0, l2=0)`, disables regularization.
    """
    return {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0.
        }
    }


def get_regularizer(hparams=None):
    """Returns a variable regularizer instance.

    See :func:`~texar.core.default_regularizer_hparams` for all
    hyperparameters and default values.

    The "type" field can be a subclass
    of :class:`~texar.core.regularizers.Regularizer`, its string name
    or module path, or a class instance.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters are set to default values.

    Returns:
        A :class:`~texar.core.regularizers.Regularizer` instance.
        `None` if :attr:`hparams` is `None` or taking the default
        hyperparameter value.

    Raises:
        ValueError: The resulting regularizer is not an instance of
            :class:`~texar.core.regularizers.Regularizer`.
    """
    if hparams is None:
        return None

    if isinstance(hparams, dict):
        hparams = HParams(hparams, default_regularizer_hparams())

    rgl = utils.check_or_get_instance(
        hparams.type, hparams.kwargs.todict(),
        ["texar.core.regularizers", "texar.custom"])

    if not isinstance(rgl, Regularizers.Regularizer):
        raise ValueError("The regularizer must be an instance of "
                         "texar.core.regularizer.Regularizer.")

    if isinstance(rgl, Regularizers.L1L2) and \
            rgl.l1 == 0. and rgl.l2 == 0.:
        return None

    return rgl


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


def get_activation_fn(fn_name="ReLU", kwargs=None):
    """Returns an activation function `fn` with the signature `output = fn(input)`.

    If the function specified by :attr:`fn_name` has more than one arguments
    without default values, then all these arguments except the input feature
    argument must be specified in :attr:`kwargs`. Arguments with default values
    can also be specified in :attr:`kwargs` to take values other than the
    defaults. In this case a partial function is returned with the above
    signature.

    Args:
        fn_name (str or callable): An activation function, or its name or
            module path. The function can be:

            - Built-in function defined in :torch_main:`torch.nn.modules.activation <nn/modules/activation.html>`
            - User-defined activation functions in module :mod:`texar.custom`.
            - External activation functions. Must provide the full module path,\
              e.g., "my_module.my_activation_fn".

        kwargs (optional): A `dict` or instance of :class:`~texar.HParams`
            containing the keyword arguments of the activation function.

    Returns:
        An activation function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    fn_modules = ['torch.nn.modules.activation', 'texar.custom', 'texar.core.layers']
    activation_fn_ = utils.get_function(fn_name, fn_modules)
    activation_fn = activation_fn_

    # Make a partial function if necessary
    if kwargs is not None:
        if isinstance(kwargs, HParams):
            kwargs = kwargs.todict()
        def _partial_fn(features):
            return activation_fn_(features, **kwargs)
        activation_fn = _partial_fn

    return activation_fn


def get_constraint_fn(fn_name="positive"):
    """Returns a constraint function.

    .. role:: python(code)
       :language: python

    The function must follow the signature:
    :python:`w_ = constraint_fn(w)`.

    Args:
        fn_name (str or callable): The name or full path to a
            constraint function, or the function itself.

            The function can be:

            - Built-in constraint functions defined in modules \
            :torch_main:`torch.distributions.constraints <distributions/constraints.html>`
            - User-defined function in :mod:`texar.custom`.
            - Externally defined function. Must provide the full path, \
            e.g., `"my_module.my_constraint_fn"`.

            If a callable is provided, then it is returned directly.

    Returns:
        The constraint function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    fn_modules = ['torch.distributions.constraints', 'texar.custom']
    constraint_fn = utils.get_function(fn_name, fn_modules)
    return constraint_fn


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
        layer_modules = ["torch.nn", "texar.core", "texar.custom"]
        layer_class = utils.check_or_get_class(layer_type, layer_modules)
        if isinstance(hparams, dict):
            default_kwargs = _layer_class_to_default_kwargs_map.get(layer_class,
                                                                    {})
            default_hparams = {"type": layer_type, "kwargs": default_kwargs}
            hparams = HParams(hparams, default_hparams)

        layer = utils.get_instance(layer_type, hparams.kwargs.todict(), layer_modules)

    if not isinstance(layer, torch.nn.Module):
        raise ValueError("layer must be an instance of `torch.nn.Module`.")

    return layer


def default_linear_kwargs():
    """TODO avinash: how to give suitable values to in_features and out_features"""
    kwargs = {
        "in_features": 32,
        "out_features": 64
    }

    return kwargs


# TODO avinash - change the following to pytorch equivalent
_layer_class_to_default_kwargs_map = {
    torch.nn.modules.linear.Linear: default_linear_kwargs()
}

"""tf.layers.Conv1D: default_conv1d_kwargs(),
tf.layers.Conv2D: default_conv2d_kwargs(),
tf.layers.Conv3D: default_conv3d_kwargs(),
tf.layers.Conv2DTranspose: default_conv2d_transpose_kwargs(),
tf.layers.Conv3DTranspose: default_conv3d_transpose_kwargs(),
tf.layers.Dense: default_dense_kwargs(),
tf.layers.Dropout: default_dropout_kwargs(),
tf.layers.Flatten: default_flatten_kwargs(),
tf.layers.MaxPooling1D: default_max_pooling1d_kwargs(),
tf.layers.MaxPooling2D: default_max_pooling2d_kwargs(),
tf.layers.MaxPooling3D: default_max_pooling3d_kwargs(),
tf.layers.SeparableConv2D: default_separable_conv2d_kwargs(),
tf.layers.BatchNormalization: default_batch_normalization_kwargs(),
tf.layers.AveragePooling1D: default_average_pooling1d_kwargs(),
tf.layers.AveragePooling2D: default_average_pooling2d_kwargs(),
tf.layers.AveragePooling3D: default_average_pooling3d_kwargs()"""
