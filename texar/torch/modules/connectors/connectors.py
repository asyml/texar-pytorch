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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributions.distribution import Distribution

from texar.torch.core import get_activation_fn
from texar.torch.hyperparams import HParams
from texar.torch.modules.connectors.connector_base import ConnectorBase
from texar.torch.utils import nest
from texar.torch.utils import utils
from texar.torch.utils.types import MaybeTuple

__all__ = [
    "ConstantConnector",
    "ForwardConnector",
    "MLPTransformConnector",
    "ReparameterizedStochasticConnector",
    "StochasticConnector",
    # "ConcatConnector"
]

TensorStruct = Union[List[torch.Tensor],
                     Dict[Any, torch.Tensor],
                     MaybeTuple[torch.Tensor]]
OutputSize = MaybeTuple[Union[int, torch.Size]]
ActivationFn = Callable[[torch.Tensor], torch.Tensor]
LinearLayer = Callable[[torch.Tensor], torch.Tensor]


def _assert_same_size(outputs: TensorStruct,
                      output_size: OutputSize):
    r"""Check if outputs match output_size

    Args:
        outputs: A tensor or a (nested) tuple of tensors
        output_size: Can be an ``int``, a ``torch.Size``, or a (nested)
            tuple of ``int`` or ``torch.Size``.
    """
    flat_output_size = nest.flatten(output_size)
    flat_output = nest.flatten(outputs)

    for (output, size) in zip(flat_output, flat_output_size):

        if isinstance(size, torch.Size):
            if output[0].size() != size:
                raise ValueError("The output size does not match"
                                 "the required output_size")
        elif output[0].size()[-1] != size:
            raise ValueError(
                "The output size does not match the required output_size")


def _get_sizes(sizes: List[Any]) -> List[int]:
    r"""

    Args:
        sizes: A list of ``int`` or ``torch.Size``. If each element is of type
            ``torch.Size``, the size is computed by taking the product of the
            shape.

    Returns:
        A list of sizes with ``torch.Size`` replaced by product of its
        individual dimensions
    """
    if isinstance(sizes[0], torch.Size):
        size_list = [np.prod(shape) for shape in sizes]
    else:
        size_list = sizes

    return size_list


def _sum_output_size(output_size: OutputSize) -> int:
    r"""Return sum of all dim values in :attr:`output_size`

    Args:
        output_size: Can be an ``int``, a ``torch.Size``, or a (nested)
            tuple of ``int`` or ``torch.Size``.
    """
    flat_output_size = nest.flatten(output_size)
    size_list = _get_sizes(flat_output_size)
    ret = sum(size_list)
    return ret


def _mlp_transform(inputs: TensorStruct,
                   output_size: OutputSize,
                   linear_layer: Optional[LinearLayer] = None,
                   activation_fn: Optional[ActivationFn] = None) -> Any:
    r"""Transforms inputs through a fully-connected layer that creates
    the output with specified size.

    Args:
        inputs: A Tensor of shape `[batch_size, d1, ..., dn]`, or a (nested)
            tuple of such elements. The dimensions `d1, ..., dn` will be flatten
            and transformed by a dense layer.
        output_size: Can be an ``int``, a ``torch.Size``, or a (nested)
            tuple of ``int`` or ``torch.Size``.
        activation_fn: Activation function applied to the output.

    :returns:
        If :attr:`output_size` is an ``int`` or a ``torch.Size``,
        returns a tensor of shape ``[batch_size, *, output_size]``.
        If :attr:`output_size` is a tuple of ``int`` or ``torch.Size``,
        returns a tuple having the same structure as :attr:`output_size`,
        where each element has the same size as defined in :attr:`output_size`.
    """
    # Flatten inputs
    flat_input = nest.flatten(inputs)
    flat_input = [x.view(-1, x.size(-1)) for x in flat_input]
    concat_input = torch.cat(flat_input, 1)

    # Get output dimension
    flat_output_size = nest.flatten(output_size)

    size_list = _get_sizes(flat_output_size)

    fc_output = concat_input
    if linear_layer is not None:
        fc_output = linear_layer(fc_output)
    if activation_fn is not None:
        fc_output = activation_fn(fc_output)

    flat_output = torch.split(fc_output, size_list, dim=1)
    flat_output = list(flat_output)

    if isinstance(flat_output_size[0], torch.Size):
        flat_output = [torch.reshape(output, (-1,) + shape) for output, shape
                       in zip(flat_output, flat_output_size)]

    output = nest.pack_sequence_as(structure=output_size,
                                   flat_sequence=flat_output)
    return output


class ConstantConnector(ConnectorBase):
    r"""Creates a constant tensor or (nested) tuple of Tensors that
    contains a constant value.

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set :attr:`output_size` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a tuple of ``int``, a ``torch.Size``,
            or a tuple of ``torch.Size``.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
            If :attr:`output_size` is a tuple ``(1, 2, 3)``, then the
            output structure will be
            ``([batch_size * 1], [batch_size * 2], [batch_size * 3])``.
            If :attr:`output_size` is ``torch.Size([1, 2, 3])``, then the
            output structure will be ``[batch_size, 1, 2, 3]``.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
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
            size_state = connector_size(batch_size=64)
            # `size_state` structure: Tensor with size [64, 1, 2, 3].
            # Tensor is filled with 2.0.

    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: Optional[HParams] = None):

        super().__init__(output_size, hparams=hparams)

        self.value = self.hparams.value

    @staticmethod
    def default_hparams() -> dict:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "value": 0.,
                "name": "constant_connector"
            }

        Here:

        `"value"`: float
            The constant scalar that the output tensor(s) has.
        `"name"`: str
            Name of the connector.
        """
        return {
            "value": 0.,
            "name": "constant_connector"
        }

    def forward(self,  # type: ignore
                batch_size: Union[int, torch.Tensor]) -> Any:
        r"""Creates output tensor(s) that has the given value.

        Args:
            batch_size: An ``int`` or ``int`` scalar tensor, the
                batch size.

        :returns:
            A (structure of) tensor whose structure is the same as
            :attr:`output_size`, with value specified by
            ``value`` or :attr:`hparams`.
        """

        def full_tensor(x):
            if isinstance(x, torch.Size):
                return torch.full((batch_size,) + x, self.value)
            else:
                return torch.full((batch_size, x), self.value)

        output = utils.map_structure(
            full_tensor,
            self._output_size)

        return output


class ForwardConnector(ConnectorBase):
    r"""Transforms inputs to have specified structure.

    Example:

    .. code-block:: python

        state_size = namedtuple('LSTMStateTuple', ['h', 'c'])(256, 256)
        # state_size == LSTMStateTuple(c=256, h=256)
        connector = ForwardConnector(state_size)
        output = connector([tensor_1, tensor_2])
        # output == LSTMStateTuple(c=tensor_1, h=tensor_2)

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set :attr:`output_size` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a tuple of ``int``, a ``torch.Size``, or a
            tuple of ``torch.Size``.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    This connector does not have trainable parameters.
    See :meth:`forward` for the inputs and outputs of the connector.
    The input to the connector must have the same structure with
    :attr:`output_size`, or must have the same number of elements and be
    re-packable into the structure of :attr:`output_size`. Note that if input
    is or contains a ``dict`` instance, the keys will be sorted to pack in
    deterministic order (See :func:`~texar.torch.utils.nest.pack_sequence_as`).
    """

    def __init__(self,
                 output_size: OutputSize,
                 hparams: Optional[HParams] = None):
        super().__init__(output_size, hparams=hparams)

    @staticmethod
    def default_hparams() -> dict:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "forward_connector"
            }

        Here:

        `"name"`: str
            Name of the connector.
        """
        return {
            "name": "forward_connector"
        }

    def forward(self,  # type: ignore
                inputs: TensorStruct) -> Any:
        r"""Transforms inputs to have the same structure as with
        :attr:`output_size`. Values of the inputs are not changed.
        :attr:`inputs` must either have the same structure, or have the same
        number of elements with :attr:`output_size`.

        Args:
            inputs: The input (structure of) tensor to pass forward.

        :returns:
            A (structure of) tensors that re-packs :attr:`inputs` to have
            the specified structure of :attr:`output_size`.
        """
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
            Can be an ``int``, a tuple of ``int``, a ``torch.Size``,
            or a tuple of ``torch.Size``.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
        linear_layer_dim (int): Value of final dim of the input tensors i.e. the
            input dim of the mlp linear layer.
        hparams (dict, optional): Hyperparameters. Missing hyperparameter will
            be set to default values. See :meth:`default_hparams` for the
            hyperparameter structure and default values.

    The input to the connector can have arbitrary structure and size.
    """

    def __init__(self,
                 output_size: OutputSize,
                 linear_layer_dim: int,
                 hparams: Optional[HParams] = None):
        super().__init__(output_size, hparams=hparams)
        self._linear_layer = nn.Linear(
            linear_layer_dim, _sum_output_size(output_size))
        self._activation_fn = get_activation_fn(
            self.hparams.activation_fn)

    @staticmethod
    def default_hparams() -> dict:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "activation_fn": "texar.torch.core.layers.identity",
                "name": "mlp_connector"
            }

        Here:

        `"activation_fn"`: str or callable
            The activation function applied to the outputs of the MLP
            transformation layer. Can
            be a function, or its name or module path.
        `"name"`: str
            Name of the connector.
        """
        return {
            "activation_fn": "texar.torch.core.layers.identity",
            "name": "mlp_connector"
        }

    def forward(self,  # type: ignore
                inputs: TensorStruct) -> Any:
        r"""Transforms inputs with an MLP layer and packs the results to have
        the same structure as specified by :attr:`output_size`.

        Args:
            inputs: Input (structure of) tensors to be transformed. Must be a
                tensor of shape ``[batch_size, ...]`` or a (nested)
                tuple of such Tensors. That is, the first dimension of
                (each) tensor must be the batch dimension.

        :returns:
            A tensor or a (nested) tuple of tensors of the same structure of
            :attr:`output_size`.
        """

        output = _mlp_transform(
            inputs, self._output_size,
            self._linear_layer, self._activation_fn)

        return output


class ReparameterizedStochasticConnector(ConnectorBase):
    r"""Samples from a distribution with reparameterization trick, and
    transforms samples into specified size.
    Reparameterization allows gradients to be back-propagated through the
    stochastic samples. Used in, e.g., Variational Autoencoders (VAEs).

    Example:

        .. code-block:: python

            # Initialized without num_samples
            cell = LSTMCell(num_units=256)
            # cell.state_size == LSTMStateTuple(c=256, h=256)
            mu = torch.zeros([16, 100])
            var = torch.ones([100])

            connector = ReparameterizedStochasticConnector(
                cell.state_size,
                mlp_input_size=mu.size()[-1],
                distribution="MultivariateNormal",
                distribution_kwargs={
                    "loc": mu,
                    "scale_tril": torch.diag(var)})
            output, sample = connector()
            # output == LSTMStateTuple(c=tensor_of_shape_(16, 256),
            #                          h=tensor_of_shape_(16, 256))
            # sample == Tensor([16, 100])

            output_, sample_ = connector(num_samples=4)
            # output_ == LSTMStateTuple(c=tensor_of_shape_(4, 16, 256),
            #                           h=tensor_of_shape_(4, 16, 256))
            # sample == Tensor([4, 16, 100])

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set ``output_size`` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a tuple of ``int``, a ``torch.Size``, or
            a tuple of ``torch.Size``.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
        mlp_input_size: Size of MLP transfer process input, which is equal to
            the distribution result size **excluding** the batch dimension,
            Can be ``int`` or ``torch.Size`` or a tuple of ``int``.
        distribution: A instance or name ``str`` of subclass of
            :torch:`distributions.distribution.Distribution`,
            Can be a distribution class instance or ``str``.
        distribution_kwargs (dict, optional): ``dict`` of keyword arguments
            for the :attr:`distribution`. Its keys are `str`, which are names
            of keyword arguments; Its values are corresponding values for each
            argument.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.
    """

    def __init__(self,
                 output_size: OutputSize,
                 mlp_input_size: Union[torch.Size, MaybeTuple[int], int],
                 distribution: Union[Distribution, str] = 'MultivariateNormal',
                 distribution_kwargs: Optional[Dict[str, Any]] = None,
                 hparams: Optional[HParams] = None):
        super().__init__(output_size, hparams=hparams)
        if distribution_kwargs is None:
            distribution_kwargs = {}
        self._dstr_type = distribution
        self._dstr_kwargs = distribution_kwargs

        for dstr_attr, dstr_val in distribution_kwargs.items():
            if isinstance(dstr_val, torch.Tensor):
                dstr_param = nn.Parameter(dstr_val)
                distribution_kwargs[dstr_attr] = dstr_param
                self.register_parameter(dstr_attr, dstr_param)
        if isinstance(mlp_input_size, int):
            input_feature = mlp_input_size
        else:
            input_feature = np.prod(mlp_input_size)
        self._linear_layer = nn.Linear(
            input_feature, _sum_output_size(output_size))

        self._activation_fn = get_activation_fn(
            self.hparams.activation_fn)

    @staticmethod
    def default_hparams() -> dict:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "activation_fn": "texar.torch.core.layers.identity",
                "name": "reparameterized_stochastic_connector"
            }

        Here:

        `"activation_fn"`: str
            The activation function applied to the outputs of the MLP
            transformation layer. Can be a function, or its name or module path.
        `"name"`: str
            Name of the connector.
        """
        return {
            "activation_fn": "texar.torch.core.layers.identity",
            "name": "reparameterized_stochastic_connector"
        }

    def forward(self,  # type: ignore
                num_samples: Optional[Union[int, torch.Tensor]] = None,
                transform: bool = True) -> Tuple[Any, Any]:
        r"""Samples from a distribution and optionally performs transformation
        with an MLP layer.
        The distribution must be reparameterizable, i.e.,
        :python:`Distribution.has_rsample == True`.

        Args:
            num_samples (optional): An ``int`` or ``int`` tensor.
                Number of samples to generate. If not given,
                generate a single sample. Note that if batch size has
                already been included in :attr:`distribution`'s dimensionality,
                :attr:`num_samples` should be left as ``None``.
            transform (bool): Whether to perform MLP transformation of the
                distribution samples. If ``False``, the structure/shape of a
                sample must match :attr:`output_size`.


        :returns:
            A tuple (:attr:`output`, :attr:`sample`), where

            - output: A tensor or a (nested) tuple of Tensors with
              the same structure and size of :attr:`output_size`.
              The batch dimension equals :attr:`num_samples` if specified,
              or is determined by the distribution dimensionality.
              If :attr:`transform` is `False`, it will be
              equal to :attr:`sample`.
            - sample: The sample from the distribution, prior to transformation.

            Otherwise, returns a tensor :attr:`sample`, where
            - sample: The sample from the distribution, prior to transformation.

        Raises:
            ValueError: If distribution is not reparameterizable.
            ValueError: The output does not match :attr:`output_size`.
        """
        if isinstance(self._dstr_type, str):
            dstr: Distribution = utils.check_or_get_instance(
                self._dstr_type, self._dstr_kwargs,
                ["torch.distributions", "texar.torch.custom"])
        else:
            dstr = self._dstr_type

        if not dstr.has_rsample:
            raise ValueError("Distribution should be reparameterizable")

        if num_samples:
            sample = dstr.rsample([num_samples])
        else:
            sample = dstr.rsample()

        if transform:
            output = _mlp_transform(
                sample,
                self._output_size,
                self._linear_layer,
                self._activation_fn)
            _assert_same_size(output, self._output_size)

        else:
            output = sample

        return output, sample


class StochasticConnector(ConnectorBase):
    r"""Samples from a distribution and transforms samples into specified size.
    The connector is the same as
    :class:`~texar.torch.modules.ReparameterizedStochasticConnector`, except
    that here reparameterization is disabled, and thus the gradients cannot be
    back-propagated through the stochastic samples.

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set ``output_size`` to ``dim`` to generate output of
            shape ``[batch_size, dim]``.
            Can be an ``int``, a tuple of ``int``, a torch.Size, or a tuple of
            torch.Size.
            For example, to transform inputs to have decoder state size, set
            :python:`output_size=decoder.state_size`.
        mlp_input_size: Size of MLP transfer process input, which is equal to
            the distribution result size **excluding** the batch dimension,
            Can be ``int`` or ``torch.Size`` or a tuple of ``int``.
        distribution: A instance of subclass of
            :torch:`distributions.distribution.Distribution`,
            Can be a class, its name or module path, or a class instance.
            The :attr:`distribution` should not be reparameterizable.
        distribution_kwargs (dict, optional): ``dict`` of keyword arguments
            for the :attr:`distribution`. Its keys are `str`, which are names
            of keyword arguments; Its values are corresponding values for each
            argument.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.
    """

    def __init__(self,
                 output_size: OutputSize,
                 mlp_input_size: Union[torch.Size, MaybeTuple[int], int],
                 distribution: Union[Distribution, str] = 'MultivariateNormal',
                 distribution_kwargs: Optional[Dict[str, Any]] = None,
                 hparams: Optional[HParams] = None):
        super().__init__(output_size, hparams=hparams)
        if distribution_kwargs is None:
            distribution_kwargs = {}
        self._dstr_kwargs = distribution_kwargs
        if isinstance(distribution, str):
            self._dstr: Distribution = utils.check_or_get_instance(
                distribution, self._dstr_kwargs,
                ["torch.distributions", "texar.torch.custom"])
        else:
            self._dstr = distribution

        if self._dstr.has_rsample:
            raise ValueError("Distribution should not be reparameterizable")

        if isinstance(mlp_input_size, int):
            input_feature = mlp_input_size
        else:
            input_feature = np.prod(mlp_input_size)
        self._linear_layer = nn.Linear(
            input_feature, _sum_output_size(output_size))

        self._activation_fn = get_activation_fn(
            self.hparams.activation_fn)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "activation_fn": "texar.torch.core.layers.identity",
                "name": "stochastic_connector"
            }

        Here:

        `"activation_fn"`: str
            The activation function applied to the outputs of the MLP
            transformation layer. Can
            be a function, or its name or module path.
        `"name"`: str
            Name of the connector.
        """
        return {
            "activation_fn": "texar.torch.core.layers.identity",
            "name": "stochastic_connector"
        }

    def forward(self,  # type: ignore
                num_samples: Optional[Union[int, torch.Tensor]] = None,
                transform: bool = False) -> Any:
        r"""Samples from a distribution and optionally performs transformation
        with an MLP layer.
        The inputs and outputs are the same as
        :class:`~texar.torch.modules.ReparameterizedStochasticConnector` except
        that the distribution does not need to be reparameterizable, and
        gradient cannot be back-propagate through the samples.

        Args:
            num_samples (optional): An ``int`` or ``int`` tensor.
                Number of samples to generate. If not given,
                generate a single sample. Note that if batch size has
                already been included in :attr:`distribution`'s dimensionality,
                :attr:`num_samples` should be left as ``None``.
            transform (bool): Whether to perform MLP transformation of the
                distribution samples. If ``False``, the structure/shape of a
                sample must match :attr:`output_size`.

        :returns:
            A tuple (:attr:`output`, :attr:`sample`), where

            - output: A tensor or a (nested) tuple of Tensors with
              the same structure and size of :attr:`output_size`.
              The batch dimension equals :attr:`num_samples` if specified,
              or is determined by the distribution dimensionality.
              If :attr:`transform` is `False`, it will be
              equal to :attr:`sample`.
            - sample: The sample from the distribution, prior to transformation.

        Raises:

            ValueError: If distribution can be reparameterizable.
            ValueError: The output does not match :attr:`output_size`.
        """

        if num_samples:
            sample = self._dstr.sample([num_samples])
        else:
            sample = self._dstr.sample()

        if self._dstr.event_shape == []:
            sample = torch.reshape(
                input=sample, shape=sample.size() + torch.Size([1]))

        # Disable gradients through samples
        sample = sample.detach().float()
        if transform:
            output = _mlp_transform(
                sample,
                self._output_size,
                self._linear_layer,
                self._activation_fn)
            _assert_same_size(output, self._output_size)

        else:
            output = sample

        return output, sample

# class ConcatConnector(ConnectorBase):
#    r"""Concatenates multiple connectors into one connector. Used in, e.g.,
#    semi-supervised variational autoencoders, disentangled representation
#    learning, and other models.
#
#    Args:
#        output_size: Size of output excluding the batch dimension (eg.
#            :attr:`output_size = p` if :attr:`output.shape` is :attr:`[N, p]`).
#            Can be an int, a tuple of int, a torch.Size, or a tuple of
#            torch.Size.
#            For example, to transform to decoder state size, set
#            `output_size=decoder.cell.state_size`.
#        hparams (dict): Hyperparameters of the connector.
#    """
#
#    def __init__(self, output_size, hparams=None):
#        super().__init__(self, output_size, hparams)
#
#    @staticmethod
#    def default_hparams():
#        r"""Returns a dictionary of hyperparameters with default values.
#
#        Returns:
#
#            .. code-block:: python
#
#                {
#                    "activation_fn": "texar.torch.core.layers.identity",
#                    "name": "concat_connector"
#                }
#
#            Here:
#
#            `"activation_fn"`: (str or callable)
#                The name or full path to the activation function applied to
#                the outputs of the MLP layer. The activation functions can be:
#
#                - Built-in activation functions defined in :
#                - User-defined activation functions in `texar.torch.custom`.
#                - External activation functions. Must provide the full path, \
#                  e.g., "my_module.my_activation_fn".
#
#                The default value is :attr:`"identity"`, i.e., the MLP
#                transformation is linear.
#
#            `"name"`: str
#                Name of the connector.
#
#                The default value is "concat_connector".
#        """
#        return {
#            "activation_fn": "texar.torch.core.layers.identity",
#            "name": "concat_connector"
#        }
#
#    def forward(self, connector_inputs, transform=True):
#        r"""Concatenate multiple input connectors
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
#            fn_modules = ['texar.torch.custom', 'torch', 'torch.nn']
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
