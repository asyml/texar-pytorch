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

from typing import Optional, Union, Callable, Tuple, List
import numpy as np

import torch

from texar import HParams
from texar.modules.connectors.connector_base import ConnectorBase
from texar.core import layers
from texar.utils.utils import get_function, check_or_get_instance
from texar.utils import nest

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

OutputSize = Union[int, torch.Size, Tuple["OutputSize"]]
HParamsType = Optional[Union[HParams, dict]]
ActivationFn = Callable[[torch.Tensor], torch.Tensor]
TensorOrNestedTuple = Union[torch.Tensor, Tuple["TensorOrNestedTuple"]]

def _assert_same_size(outputs: TensorOrNestedTuple,
                      output_size: OutputSize):
    """Check if outputs match output_size
    Args:
        outputs: A Tensor or a (nested) tuple of tensors
        output_size: Can be an Integer, a torch.Size, or a (nested) tuple of
            Integers or torch.Size.
    """
    nest.assert_same_structure(outputs, output_size)
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
    """Returns the size of a tensor excluding the first dimension
    (typically the batch dimension).

    Args:
        x: A tensor.
    """
    return np.prod(x.size()[1:])

def _mlp_transform(inputs: TensorOrNestedTuple,
                   output_size: OutputSize,
                   activation_fn: ActivationFn = layers.identity
                   ) -> TensorOrNestedTuple:
    """Transforms inputs through a fully-connected layer that creates the output
    with specified size.

    Args:
        inputs: A Tensor of shape `[batch_size, ...]` (i.e., batch-major), or a
            (nested) tuple of such elements. A Tensor or a (nested) tuple of
            Tensors with shape `[max_time, batch_size, ...]` (i.e., time-major)
            can be transposed to batch-major using
            :func:`~texar.utils.transpose_batch_time` prior to this
            function.
        output_size: Can be an Integer, a torch.Size, or a (nested) tuple of
            Integers or torch.Size.
        activation_fn: Activation function applied to the output.

    Returns:
        If :attr:`output_size` is an Integer or a torch.Size, returns a Tensor
        of shape `[batch_size x output_size]`. If :attr:`output_size` is a tuple
        of Integers or TensorShape, returns a tuple having the same structure as
        :attr:`output_size`, where each element Tensor has the same size as
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
    fc_output = torch.nn.Linear(
        concat_input.size()[-1], sum_output_size)(concat_input)
    fc_output = activation_fn(fc_output)
    flat_output = torch.split(fc_output, size_list, dim=1)
    flat_output = list(flat_output)
    if isinstance(flat_output_size[0], torch.Size):
        for (i, shape) in enumerate(flat_output_size):
            flat_output[i] = torch.reshape(
                flat_output[i], tuple([-1] + list(shape)))

    output = nest.pack_sequence_as(structure=output_size,
                                   flat_sequence=flat_output)
    return output


class ConstantConnector(ConnectorBase):
    """Creates a constant Tensor or (nested) tuple of Tensors that
    contains a constant value.

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set `output_size` to `dim` to generate output of
            shape `[batch_size, dim]`.
            Can be an `int`, a tuple of `int`, a torch.Size, or a tuple of
            torch.Size.
            For example, to transform inputs to have decoder state size, set
            `output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    This connector does not have trainable parameters.
    See :meth:`_build` for the inputs and outputs of the connector.

    Example:

        .. code-block:: python

            connector = Connector(cell.state_size)
            zero_state = connector(batch_size=64, value=0.)
            one_state = connector(batch_size=64, value=1.)

    .. document private functions
    .. automethod:: _build
    """
    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None):

        ConnectorBase.__init__(self, output_size, hparams)
        self._built = False

    @staticmethod
    def default_hparams() -> dict:
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "value": 0.,
                "name": "constant_connector"
            }

        Here:

        "value" : float
            The constant scalar that the output tensor(s) has. Ignored if
            `value` is given to :meth:`_build`.

        "name" : str
            Name of the connector.
        """
        return {
            "value": 0.,
            "name": "constant_connector"
        }

    def forward(self,
                batch_size: Union[int, torch.Tensor],
                value: Optional = None) -> TensorOrNestedTuple:
        """Creates output tensor(s) that has the given value.

        Args:
            batch_size: An `int` or `int` scalar Tensor, the batch size.
            value (optional): A scalar, the value that
                the output tensor(s) has. If `None`, "value" in :attr:`hparams`
                is used.

        Returns:
            A (structure of) tensor whose structure is the same as
            :attr:`output_size`, with value speicified by
            `value` or :attr:`hparams`.
        """
        value_ = value
        if value_ is None:
            value_ = self.hparams.value
        output = nest.map_structure(
            lambda x: torch.full((batch_size, x), value_),
            self._output_size)

        self._built = True

        return output


class ForwardConnector(ConnectorBase):
    """Transforms inputs to have specified structure.

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set `output_size` to `dim` to generate output of
            shape `[batch_size, dim]`.
            Can be an `int`, a tuple of `int`, a torch.Size, or a tuple of
            torch.Size.
            For example, to transform inputs to have decoder state size, set
            `output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    This connector does not have trainable parameters.
    See :meth:`forward` for the inputs and outputs of the connector.

    The input to the connector must have the same structure with
    :attr:`output_size`, or must have the same number of elements and be
    re-packable into the structure of :attr:`output_size`. Note that if input
    is or contains a `dict` instance, the keys will be sorted to pack in
    deterministic order (See
    `pack_sequence_as <texar/texar/utils/nest.py>`
    for more details).

    Example:

        .. code-block:: python

            cell = LSTMCell(num_units=256)
            # cell.state_size == LSTMStateTuple(c=256, h=256)

            connector = ForwardConnector(cell.state_size)
            output = connector([tensor_1, tensor_2])
            # output == LSTMStateTuple(c=tensor_1, h=tensor_2)

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None):
        ConnectorBase.__init__(self, output_size, hparams)
        self._built = False

    @staticmethod
    def default_hparams() -> dict:
        """Returns a dictionary of hyperparameters with default values.

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

    def forward(self,
                inputs: Union[list, dict, tuple, torch.Tensor]
                ) -> TensorOrNestedTuple:
        """Transforms inputs to have the same structure as with
        :attr:`output_size`. Values of the inputs are not changed.

        :attr:`inputs` must either have the same structure, or have the same
        number of elements with :attr:`output_size`.

        Args:
            inputs: The input (structure of) tensor to pass forward.

        Returns:
            A (structure of) tensors that re-packs `inputs` to have
            the specified structure of `output_size`.
        """
        output = inputs
        try:
            nest.assert_same_structure(inputs, self._output_size)
        except (ValueError, TypeError):
            flat_input = nest.flatten(inputs)
            output = nest.pack_sequence_as(
                self._output_size, flat_input)

        self._built = True

        return output


class MLPTransformConnector(ConnectorBase):
    """Transforms inputs with an MLP layer and packs the results into the
    specified structure and size.

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set `output_size` to `dim` to generate output of
            shape `[batch_size, dim]`.
            Can be an `int`, a tuple of `int`, a Tensorshape, or a tuple of
            torch.Size.
            For example, to transform inputs to have decoder state size, set
            `output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`_build` for the inputs and outputs of the connector.

    The input to the connector can have arbitrary structure and size.

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

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None):
        ConnectorBase.__init__(self, output_size, hparams)
        self._built = False

    @staticmethod
    def default_hparams() -> dict:
        """Returns a dictionary of hyperparameters with default values.

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

    def forward(self,
                inputs: TensorOrNestedTuple
                ) -> TensorOrNestedTuple:
        """Transforms inputs with an MLP layer and packs the results to have
        the same structure as specified by :attr:`output_size`.

        Args:
            inputs: Input (structure of) tensors to be transformed. Must be a
                Tensor of shape `[batch_size, ...]` or a (nested) tuple of
                such Tensors. That is, the first dimension of (each) tensor
                must be the batch dimension.

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            `output_size`.
        """
        activation_fn = layers.get_activation_fn(self.hparams.activation_fn)

        output = _mlp_transform(inputs, self._output_size, activation_fn)

        self._built = True

        return output


class ReparameterizedStochasticConnector(ConnectorBase):
    """Samples from a distribution with reparameterization trick, and
    transforms samples into specified size.

    Reparameterization allows gradients to be back-propagated through the
    stochastic samples. Used in, e.g., Variational Autoencoders (VAEs).

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set `output_size` to `dim` to generate output of
            shape `[batch_size, dim]`.
            Can be an `int`, a tuple of `int`, a torch.Size, or a tuple of
            torch.Size.
            For example, to transform inputs to have decoder state size, set
            `output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

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

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None):
        ConnectorBase.__init__(self, output_size, hparams)

    @staticmethod
    def default_hparams() -> dict:
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "activation_fn": "texar.core.layers.identity",
                "name": "reparameterized_stochastic_connector"
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
            "name": "reparameterized_stochastic_connector"
        }

    def forward(self,
                distribution: Union[object, str] = 'MultivariateNormal',
                distribution_kwargs: Optional[dict] = None,
                transform: bool = True,
                num_samples: Optional[Union[int, torch.Tensor]] = None
                ) -> Tuple[Union[TensorOrNestedTuple, torch.Tensor]]:
        """Samples from a distribution and optionally performs transformation
        with an MLP layer.

        The distribution must be reparameterizable, i.e.,
        `distribution.reparameterization_type = FULLY_REPARAMETERIZED`.

        Args:
            distribution: A instance of subclass of
                `Torch Distribution <distributions/Distribution>`,
                Can be a class, its name or module path, or a class instance.
            distribution_kwargs (dict, optional): Keyword arguments for the
                distribution constructor. Ignored if `distribution` is a
                class instance.
            transform (bool): Whether to perform MLP transformation of the
                distribution samples. If `False`, the structure/shape of a
                sample must match :attr:`output_size`.
            num_samples (optional): An `int` or `int` Tensor. Number of samples
                to generate. If not given, generate a single sample. Note
                that if batch size has already been included in
                `distribution`'s dimensionality, `num_samples` should be
                left as `None`.

        Returns:
            A tuple (output, sample), where

            - output: A Tensor or a (nested) tuple of Tensors with the same \
            structure and size of :attr:`output_size`. The batch dimension \
            equals :attr:`num_samples` if specified, or is determined by the \
            distribution dimensionality.
            - sample: The sample from the distribution, prior to transformation.

        Raises:
            ValueError: If distribution cannot be reparametrized.
            ValueError: The output does not match :attr:`output_size`.
        """
        dstr = check_or_get_instance(
            distribution, distribution_kwargs,
            ["torch.distributions", "torch.distributions.multivariate_normal",
             "texar.custom"])

        if num_samples:
            sample = dstr.sample([num_samples])
        else:
            sample = dstr.sample()

        if dstr.event_shape == []:
            sample = torch.reshape(
                sample,
                sample.size() + torch.Size([1]))

        sample = sample.float()
        if transform:
            fn_modules = ['torch', 'torch.nn', 'texar.custom']
            activation_fn = get_function(self.hparams.activation_fn, fn_modules)
            output = _mlp_transform(sample, self._output_size, activation_fn)

        _assert_same_size(output, self._output_size)

        return (output, sample)


class StochasticConnector(ConnectorBase):
    """Samples from a distribution and transforms samples into specified size.

    The connector is the same as
    :class:`~texar.modules.ReparameterizedStochasticConnector`, except that
    here reparameterization is disabled, and thus the gradients cannot be
    back-propagated through the stochastic samples.

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set `output_size` to `dim` to generate output of
            shape `[batch_size, dim]`.
            Can be an `int`, a tuple of `int`, a Tensorshape, or a tuple of
            torch.Size.
            For example, to transform inputs to have decoder state size, set
            `output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: HParamsType = None):
        ConnectorBase.__init__(self, output_size, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

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

    def forward(self,
                distribution: Union[object, str] = 'MultivariateNormal',
                distribution_kwargs: Optional[dict] = None,
                transform: bool = False,
                num_samples: Optional[Union[int, torch.Tensor]] = None
                ) -> TensorOrNestedTuple:
        """Samples from a distribution and optionally performs transformation
        with an MLP layer.

        The inputs and outputs are the same as
        :class:`~texar.modules.ReparameterizedStochasticConnector` except that
        the distribution does not need to be reparameterizable, and gradient
        cannot be back-propagate through the samples.

        Args:
            distribution: A instance of subclass of
                `Torch Distribution <distributions/Distribution>`,
                Can be a class, its name or module path, or a class instance.
            distribution_kwargs (dict, optional): Keyword arguments for the
                distribution constructor. Ignored if `distribution` is a
                class instance.
            transform (bool): Whether to perform MLP transformation of the
                distribution samples. If `False`, the structure/shape of a
                sample must match :attr:`output_size`.
            num_samples (optional): An `int` or `int` Tensor. Number of samples
                to generate. If not given, generate a single sample. Note
                that if batch size has already been included in
                `distribution`'s dimensionality, `num_samples` should be
                left as `None`.

        Returns:
            A Tensor or a (nested) tuple output, where

            - output: A Tensor or a (nested) tuple of Tensors with the same \
            structure and size of :attr:`output_size`. The batch dimension \
            equals :attr:`num_samples` if specified, or is determined by the \
            distribution dimensionality.

        Raises:
            ValueError: The output does not match :attr:`output_size`.
        """
        dstr = check_or_get_instance(
            distribution, distribution_kwargs,
            ["torch.distributions", "torch.distributions.multivariate_normal",
             "texar.custom"])

        if num_samples:
            output = dstr.sample([num_samples])
        else:
            output = dstr.sample()

        if dstr.event_shape == []:
            output = torch.reshape(output,
                                   output.size() + torch.Size(1))

        # Disable gradients through samples
        output = output.detach()

        output = output.float()

        if transform:
            fn_modules = ['torch', 'torch.nn', 'texar.custom']
            activation_fn = get_function(
                self.hparams.activation_fn, fn_modules)
            output = _mlp_transform(
                output, self._output_size, activation_fn)

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
#            activation_fn = get_function(self.hparams.activation_fn,
#                                         fn_modules)
#            output = _mlp_transform(output, self._output_size, activation_fn)
#        _assert_same_size(output, self._output_size)
#
#        self._add_internal_trainable_variables()
#        self._built = True
#
#        return output
