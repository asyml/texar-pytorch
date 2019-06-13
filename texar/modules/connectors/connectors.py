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
Various connectors.
"""

from typing import (Optional, Union, Callable, Tuple, List,
                    Any, Dict, Type, TypeVar)
import numpy as np

import torch
from torch import nn, split
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.distribution import Distribution

from texar.hyperparams import HParams
from texar.modules.connectors.connector_base import ConnectorBase
from texar.core import layers
from texar.utils import utils
from texar.utils import nest
from texar.utils.types import MaybeTuple


# pylint: disable=too-many-locals, arguments-differ
# pylint: disable=too-many-arguments, invalid-name, no-member

__all__ = [
    "ConstantConnector",
    "ForwardConnector",
    "MLPTransformConnector",
    "ReparameterizedStochasticConnector",
    "StochasticConnector",
    #"ConcatConnector"
]

T = TypeVar('T')
Inputs = Union[List[Any], Dict[Any, Any], Tuple[Any], torch.Tensor]
D = MultivariateNormal
OutputSize = Union[MaybeTuple[int], MaybeTuple[torch.Size], Tuple]
HParamsType = Optional[HParams]
#Optional[Union[HParams, dict]]
ActivationFn = Optional[Callable[[torch.Tensor], torch.Tensor]]
LinearLayer = Callable[[torch.Tensor], torch.Tensor]

def _assert_same_size(outputs: Inputs,
                      output_size: OutputSize):
    r"""Check if outputs match output_size

    Args:
        outputs: A ``Tensor`` or a (nested) ``tuple`` of tensors
        output_size: Can be an Integer, a ``torch.Size``, or a (nested)
            ``tuple`` of Integers or ``torch.Size``.
    """
    flat_output_size = nest.flatten(output_size)
    flat_output = nest.flatten(outputs)

    for (output, size) in zip(flat_output, flat_output_size):

        if isinstance(size, torch.Size):
            if output[0].size() != size:
                raise ValueError("The output size does not match"
                                 "the the required output_size")
        elif output[0].size() != torch.Size([size]):
            raise ValueError(
                "The output size does not match the the required output_size")

def _get_tensor_depth(x: torch.Tensor) -> int:
    r"""Returns the size of a tensor excluding the first dimension
    (typically the batch dimension).

    Args:
        x: A tensor.
    """
    return int(np.prod(x.size()[1:]))

def _mlp_transform(inputs: Inputs,
                   output_size: OutputSize,
                   activation_fn: ActivationFn = layers.identity,
                   ) -> Any:
    r"""Transforms inputs through a fully-connected layer that creates
    the output with specified size.

    Args:
        inputs: A ``Tensor`` of shape ``[batch_size, ...]``
            (i.e., batch-major), or a (nested) tuple of such elements.
            A Tensor or a (nested) tuple of Tensors with shape
            ``[max_time, batch_size, ...]`` (i.e., time-major) can
            be transposed to batch-major using
            :func:`~texar.utils.transpose_batch_time` prior to this function.
        output_size: Can be an ``Integer``, a ``torch.Size``, or a (nested)
            ``tuple`` of ``Integers`` or ``torch.Size``.
        activation_fn: Activation function applied to the output.

    Returns:
        If :attr:`output_size` is an ``Integer`` or a ``torch.Size``,
        returns a ``Tensor`` of shape ``[batch_size x output_size]``.
        If :attr:`output_size` is a ``tuple`` of Integers or TensorShape,
        returns a ``tuple`` having the same structure as:attr:`output_size`,
        where each element ``Tensor`` has the same size as
        defined in :attr:`output_size`.
    """
    # Flatten inputs
    flat_input = nest.flatten(inputs)
    dims = [_get_tensor_depth(x) for x in flat_input]
    flat_input = [torch.reshape(x, (-1, d)) for x, d in zip(flat_input, dims)]
    concat_input = torch.cat(flat_input, 1)

    # Get output dimension
    flat_output_size = nest.flatten(output_size)

    if isinstance(flat_output_size[0], torch.Size):
        size_list = [0] * len(flat_output_size)
        for (i, shape) in enumerate(flat_output_size):
            size_list[i] = np.prod([dim for dim in shape])
    else:
        size_list = flat_output_size

    sum_output_size = sum(size_list)
    linear_layer = nn.Linear(
        concat_input.size()[-1], sum_output_size)

    fc_output = linear_layer(concat_input)
    if activation_fn:
        fc_output = activation_fn(fc_output)

    flat_output = split(fc_output, size_list, dim=1) # type: ignore
    flat_output = list(flat_output)
    if isinstance(flat_output_size[0], torch.Size):
        for (i, shape) in enumerate(flat_output_size):
            flat_output[i] = torch.reshape(
                flat_output[i], tuple([-1] + list(shape)))

    output = nest.pack_sequence_as(structure=output_size,
                                   flat_sequence=flat_output)
    return output, linear_layer


class ConstantConnector(ConnectorBase):
    r"""Creates a constant ``Tensor`` or (nested) ``tuple`` of Tensors that
    contains a constant value. 

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set :attr:`output_size` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a tuple of ``int``, a ``torch.Size``,
            or a ``tuple`` of ``torch.Size``.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
            If :attr:`output_size` is ``tuple`` ```(1, 2, 3)```, then the
            output structure will be
            ``([batch_size * 1], [batch_size * 2], [batch_size * 3])``.
            If :attr:`output_size` is ``torch.Size([1, 2, 3])``, then the
            output structure will be ``[batch_size, 1, 2, 3]``.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    This connector does not have trainable parameters.

    Example:

        .. code-block:: python

            state_size = (1, 2, 3)
            connector = ConstantConnector(state_size, hparams={"value": 1.})
            one_state = connector(batch_size=64)
            # `one_state` structure: (Tensor_1, Tensor_2, Tensor_3),
            # Tensor_1.size() == torch.Size([64, 1])
            # Tensor_2.size() == torch.Size([64, 2])
            # Tensor_3.size() == torch.Size([64, 3])
            # Tensors are filled with 1.0.
            size = torch.Size([1, 2, 3])
            connector_size = ConstantConnector(size, hparams={"value": 2.})
            size_state = connector_size_0(batch_size=64)
            # `size_state` structure: Tensor with size [64, 1, 2, 3].
            # Tensor is filled with 2.0.

    """
    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None):

        ConnectorBase.__init__(self, output_size, hparams)

    @staticmethod
    def default_hparams() -> dict:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "value": 0.,
                "name": "constant_connector"
            }

        Here:

        "value" : float
            The constant scalar that the output tensor(s) has.
        "name" : str
            Name of the connector.
        """
        return {
            "value": 0.,
            "name": "constant_connector"
        }

    def forward(self, # type: ignore
                batch_size: Union[int, torch.Tensor]) -> Any:
        """Creates output tensor(s) that has the given value.

        Args:
            batch_size: An ``int`` or ``int`` scalar ``Tensor``, the
                batch size.
            value (optional): A scalar, the value that the output tensor(s)
                has. If ``None``, ``"value"`` in :attr:`hparams` is used.

        Returns:
            A (structure of) ``Tensor`` whose structure is the same as
            :attr:`output_size`, with value speicified by
            ``value`` or :attr:`hparams`.
        """
        value_ = self.hparams.value

        def full_tensor(x):
            if isinstance(x, torch.Size):
                return torch.full((batch_size,) + x, value_)
            else:
                return torch.full((batch_size, x), value_)

        output = utils.map_structure(
            full_tensor,
            self._output_size)

        return output


class ForwardConnector(ConnectorBase):
    """Transforms inputs to have specified structure.

    Example:

    .. code-block:: python

        state_size = namedtuple('LSTMStateTuple', ['c', 'h'])(256, 256)
        # state_size == LSTMStateTuple(c=256, h=256)
        connector = ForwardConnector(state_size)
        output = connector([tensor_1, tensor_2])
        # output == LSTMStateTuple(c=tensor_1, h=tensor_2)

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set :attr:`output_size` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a tuple of ``int``, a ``torch.Size``, or a
            ``tuple`` of ``torch.Size``.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    This connector does not have trainable parameters.
    See :meth:`forward` for the inputs and outputs of the connector.
    The input to the connector must have the same structure with
    :attr:`output_size`, or must have the same number of elements and be
    re-packable into the structure of :attr:`output_size`. Note that if input
    is or contains a ``dict`` instance, the keys will be sorted to pack in
    deterministic order (See :func:`~texar.utils.nest.pack_sequence_as`
    for more details).
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None):
        ConnectorBase.__init__(self, output_size, hparams)

    @staticmethod
    def default_hparams() -> dict:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "forward_connector"
            }

        Here:

        "name" : str
            Name of the connector.
        """
        return {
            "name": "forward_connector"
        }

    def forward(self, # type: ignore
                inputs: Inputs
                ) -> Any:
        r"""Transforms inputs to have the same structure as with
        :attr:`output_size`. Values of the inputs are not changed.
        :attr:`inputs` must either have the same structure, or have the same
        number of elements with :attr:`output_size`.

        Args:
            inputs: The input (structure of) tensor to pass forward.

        Returns:
            A (structure of) tensors that re-packs :attr:`inputs` to have
            the specified structure of :attr:`output_size`.
        """
        output = inputs
        flat_input = nest.flatten(inputs)
        output = nest.pack_sequence_as(
            self._output_size, flat_input)

        return output


class MLPTransformConnector(ConnectorBase):
    r"""Transforms inputs with an MLP layer and packs the results into the
    specified structure and size.

    Example:

        .. code-block:: python

            cell = LSTMCell(num_units=256)
            # cell.state_size == LSTMStateTuple(c=256, h=256)
            connector = MLPTransformConnector(cell.state_size)
            inputs = torch.zeros([64, 10])
            output = connector(inputs)
            # output == LSTMStateTuple(c=tensor_of_shape_(64, 256),
            #                          h=tensor_of_shape_(64, 256))

        .. code-block:: python

            ## Use to connect encoder and decoder with different state size
            encoder = UnidirectionalRNNEncoder(...)
            _, final_state = encoder(inputs=...)
            decoder = BasicRNNDecoder(...)
            connector = MLPTransformConnector(decoder.state_size)
            _ = decoder(
                initial_state=connector(final_state),
                ...)

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set :attr:`output_size` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a ``tuple`` of ``int``, a ``torch.Size``,
            or a ``tuple`` of ``torch.Size``.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    The input to the connector can have arbitrary structure and size.
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None):
        ConnectorBase.__init__(self, output_size, hparams)

    @staticmethod
    def default_hparams() -> dict:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "activation_fn": "texar.core.layers.identity",
                "name": "mlp_connector"
            }

        Here:

        "activation_fn" : str or callable
            The activation function applied to the outputs of the MLP
            transformation layer. Can
            be a function, or its name or module path.
        "name" : str
            Name of the connector.
        """
        return {
            "activation_fn": "texar.core.layers.identity",
            "name": "mlp_connector"
        }

    def forward(self, # type: ignore
                inputs: Inputs
                ) -> Any:
        r"""Transforms inputs with an MLP layer and packs the results to have
        the same structure as specified by :attr:`output_size`.

        Args:
            inputs: Input (structure of) tensors to be transformed. Must be a
                ``Tensor`` of shape ``[batch_size, ...]`` or a (nested)
                ``tuple`` of such Tensors. That is, the first dimension of
                (each) ``Tensor`` must be the batch dimension.

        Returns:
            A ``Tensor`` or a (nested) ``tuple`` of Tensors of the
            same structure of :attr:`output_size`.
        """
        activation_fn = layers.get_activation_fn(self.hparams.activation_fn)

        output, linear_layer = _mlp_transform(
            inputs, self._output_size, activation_fn)
        self._linear_layers.append(linear_layer)

        return output


class ReparameterizedStochasticConnector(ConnectorBase):
    r"""Samples from a distribution with reparameterization trick, and
    transforms samples into specified size.
    Reparameterization allows gradients to be back-propagated through the
    stochastic samples. Used in, e.g., Variational Autoencoders (VAEs).

    Example:

        .. code-block:: python

            cell = LSTMCell(num_units=256)
            # cell.state_size == LSTMStateTuple(c=256, h=256)
            connector = ReparameterizedStochasticConnector(cell.state_size)
            kwargs = {
                'loc': torch.zeros([batch_size, 10]),
                'scale_diag': torch.stack(
                    [torch.diag(x) for x in torch.ones([batch_size, 10])],
                    0)
            }
            output, sample = connector(distribution_kwargs=kwargs)
            # output == LSTMStateTuple(c=tensor_of_shape_(batch_size, 256),
            #                          h=tensor_of_shape_(batch_size, 256))
            # sample == Tensor([batch_size, 10])
            kwargs = {
                'loc': torch.zeros([10]),
                'scale_diag': torch.stack(
                    [torch.diag(x) for x in torch.ones([10])], dim=0)
            }
            output_, sample_ = connector(distribution_kwargs=kwargs,
                                         num_samples=batch_size_)
            # output_ == LSTMStateTuple(c=tensor_of_shape_(batch_size_, 256),
            #                           h=tensor_of_shape_(batch_size_, 256))
            # sample == Tensor([batch_size_, 10])

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set ``output_size`` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a tuple of ``int``, a ``torch.Size``, or
            a ``tuple`` of ``torch.Size``.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
        distribution: A instance of subclass of
            :torch:`distributions.distribution.Distribution`,
            Can be a class, its name or module path, or a class instance.
        distribution_kwargs (dict, optional): Keyword arguments for the
            distribution constructor. Ignored if :attr:`distribution` is a
            class instance.
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None,
                 distribution: Union[Type[T], T, str] = 'MultivariateNormal',
                 distribution_kwargs: Optional[Dict[str, Any]] = None,):
        ConnectorBase.__init__(self, output_size, hparams)
        if distribution_kwargs is None:
            distribution_kwargs = {}
        self._dstr = utils.check_or_get_instance(
            distribution, distribution_kwargs,
            ["torch.distributions", "torch.distributions.multivariate_normal",
             "texar.custom"])

    @staticmethod
    def default_hparams() -> dict:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "activation_fn": "texar.core.layers.identity",
                "name": "reparameterized_stochastic_connector"
            }

        Here:

        "activation_fn" : str
            The activation function applied to the outputs of the MLP
            transformation layer. Can be a function,
            or its name or module path.
        "name" : str
            Name of the connector.
        """
        return {
            "activation_fn": "texar.core.layers.identity",
            "name": "reparameterized_stochastic_connector"
        }

    def forward(self, # type: ignore
                transform: bool = True,
                num_samples: Optional[Union[int, torch.Tensor]] = None
                ) -> Tuple[Any, Any]:
        r"""Samples from a distribution and optionally performs transformation
        with an MLP layer.
        The distribution must be reparameterizable, i.e.,
        :python:`distribution.reparameterization_type = FULLY_REPARAMETERIZED`.

        Args:
            transform (bool): Whether to perform MLP transformation of the
                distribution samples. If ``False``, the structure/shape of a
                sample must match :attr:`output_size`.
            num_samples (optional): An ``int`` or ``int`` ``Tensor``.
                Number of samples to generate. If not given,
                generate a single sample. Note that if batch size has
                already been included in :attr:`distribution`'s dimensionality,
                :attr:`num_samples` should be left as ``None``.

        Returns:
            A ``tuple`` (output, sample), where
            - output: A ``Tensor`` or a (nested) ``tuple`` of Tensors with
              the same structure and size of :attr:`output_size`.
              The batch dimension equals :attr:`num_samples` if specified,
              or is determined by the distribution dimensionality.
            - sample: The sample from the distribution, prior to transformation.
        Raises:
            ValueError: If distribution cannot be reparametrized.
            ValueError: The output does not match :attr:`output_size`.
        """

        if num_samples:
            sample = self._dstr.sample([num_samples]) # type: ignore
        else:
            sample = self._dstr.sample() # type: ignore

        if self._dstr.event_shape == []: # type: ignore
            sample = torch.reshape(
                sample,
                sample.size() + torch.Size([1]))

        sample = sample.float()
        if transform:
            fn_modules = ['torch', 'torch.nn', 'texar.custom']
            activation_fn = utils.get_function(
                self.hparams.activation_fn, fn_modules)

            output, linear_layer = _mlp_transform(
                sample, self._output_size, activation_fn)
            self._linear_layers.append(linear_layer)

            _assert_same_size(output, self._output_size)

        return (output, sample)


class StochasticConnector(ConnectorBase):
    r"""Samples from a distribution and transforms samples into specified size.
    The connector is the same as
    :class:`~texar.modules.ReparameterizedStochasticConnector`, except that
    here reparameterization is disabled, and thus the gradients cannot be
    back-propagated through the stochastic samples.

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set ``output_size`` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a tuple of ``int``, a Tensorshape, or a tuple of
            torch.Size.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
        distribution: A instance of subclass of
            :torch:`distributions.distribution.Distribution`,
            Can be a class, its name or module path, or a class instance.
        distribution_kwargs (dict, optional): Keyword arguments for the
            distribution constructor. Ignored if :attr:`distribution` is a
            class instance.
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None,
                 distribution: Union[Type[T], T, str] = 'MultivariateNormal',
                 distribution_kwargs: Optional[Dict[str, Any]] = None,):
        ConnectorBase.__init__(self, output_size, hparams)
        if distribution_kwargs is None:
            distribution_kwargs = {}
        self._dstr = utils.check_or_get_instance(
            distribution, distribution_kwargs,
            ["torch.distributions", "torch.distributions.multivariate_normal",
             "texar.custom"])

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "activation_fn": "texar.core.layers.identity",
                "name": "stochastic_connector"
            }

        Here:

        "activation_fn" : str
            The activation function applied to the outputs of the MLP
            transformation layer. Can
            be a function, or its name or module path.
        "name" : str
            Name of the connector.
        """
        return {
            "activation_fn": "texar.core.layers.identity",
            "name": "stochastic_connector"
        }

    def forward(self, # type: ignore
                transform: bool = False,
                num_samples: Optional[Union[int, torch.Tensor]] = None
                ) -> Any:
        r"""Samples from a distribution and optionally performs transformation
        with an MLP layer.
        The inputs and outputs are the same as
        :class:`~texar.modules.ReparameterizedStochasticConnector` except that
        the distribution does not need to be reparameterizable, and gradient
        cannot be back-propagate through the samples.

        Args:
            transform (bool): Whether to perform MLP transformation of the
                distribution samples. If ``False``, the structure/shape of a
                sample must match :attr:`output_size`.
            num_samples (optional): An ``int`` or ``int`` ``Tensor``. Number
                of samples to generate. If not given, generate a
                single sample. Note that if batch size has already
                been included in :attr:`distribution`'s dimensionality,
                ``num_samples`` should be left as ``None``.

        Returns:
            A ``Tensor`` or a (nested) ``tuple`` output, where
            - output: A ``Tensor`` or a (nested) ``tuple`` of Tensors
              with the same structure and size of :attr:`output_size`.
              The batch dimension equals :attr:`num_samples` if specified,
              or is determined by the distribution dimensionality.

        Raises:
            ValueError: The output does not match :attr:`output_size`.
        """

        if num_samples:
            output = self._dstr.sample([num_samples]) # type: ignore
        else:
            output = self._dstr.sample() # type: ignore

        if self._dstr.event_shape == []: # type: ignore
            output = torch.reshape(
                input=output, shape=output.size() + torch.Size([1]))
        # Disable gradients through samples
        output = output.detach()

        output = output.float()

        if transform:
            fn_modules = ['torch', 'torch.nn', 'texar.custom']
            activation_fn = utils.get_function(
                self.hparams.activation_fn, fn_modules)

            output, linear_layer = _mlp_transform(
                output, self._output_size, activation_fn)
            self._linear_layers.append(linear_layer)

        _assert_same_size(output, self._output_size)

        return output


#class ConcatConnector(ConnectorBase):
#    """Concatenates multiple connectors into one connector. Used in, e.g.,
#    semi-supervised variational autoencoders, disentangled representation
#    learning, and other models.
#
#    Args:
#        output_size: Size of output excluding the batch dimension (eg.
#            :attr:`output_size = p` if :attr:`output.shape` is :attr:`[N, p]`).
#            Can be an int, a tuple of int, a Tensorshape, or a tuple of
#            torch.Size.
#            For example, to transform to decoder state size, set
#            `output_size=decoder.cell.state_size`.
#        hparams (dict): Hyperparameters of the connector.
#    """
#
#    def __init__(self, output_size, hparams=None):
#        ConnectorBase.__init__(self, output_size, hparams)
#
#    @staticmethod
#    def default_hparams():
#        """Returns a dictionary of hyperparameters with default values.
#
#        Returns:
#
#            .. code-block:: python
#
#                {
#                    "activation_fn": "texar.core.layers.identity",
#                    "name": "concat_connector"
#                }
#
#            Here:
#
#            "activation_fn" : (str or callable)
#                The name or full path to the activation function applied to
#                the outputs of the MLP layer. The activation functions can be:
#
#                - Built-in activation functions defined in :
#                - User-defined activation functions in `texar.custom`.
#                - External activation functions. Must provide the full path, \
#                  e.g., "my_module.my_activation_fn".
#
#                The default value is :attr:`"identity"`, i.e., the MLP
#                transformation is linear.
#
#            "name" : str
#                Name of the connector.
#
#                The default value is "concat_connector".
#        """
#        return {
#            "activation_fn": "texar.core.layers.identity",
#            "name": "concat_connector"
#        }
#
#    def forward(self, connector_inputs, transform=True):
#        """Concatenate multiple input connectors
#
#        Args:
#            connector_inputs: a list of connector states
#            transform (bool): If `True`, then the output are automatically
#                transformed to match :attr:`output_size`.
#
#        Returns:
#            A Tensor or a (nested) tuple of Tensors of the same structure of
#            the decoder state.
#        """
#        connector_inputs = [connector.float()
#                            for connector in connector_inputs]
#        output = torch.cat(connector_inputs, dim=1)
#
#        if transform:
#            fn_modules = ['texar.custom', 'torch', 'torch.nn']
#            activation_fn = utils.get_function(self.hparams.activation_fn,
#                                         fn_modules)
#            output, linear_layer = _mlp_transform(
#               output, self._output_size, activation_fn)
#            self._linear_layers.append(linear_layer)
#        _assert_same_size(output, self._output_size)
#
#        self._add_internal_trainable_variables()
#        self._built = True
#
#        return output
