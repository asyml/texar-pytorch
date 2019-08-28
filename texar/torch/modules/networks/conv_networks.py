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
Various convolutional networks.
"""
from typing import List, Optional, Tuple, Union, Dict, Any

import torch

from texar.torch.core.layers import get_pooling_layer_hparams
from texar.torch.hyperparams import HParams
from texar.torch.modules.networks.network_base import FeedForwardNetworkBase
from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.utils import uniquify_str

__all__ = [
    "_to_list",
    "Conv1DNetwork",
]


def _to_list(value: Union[Dict[str, Any], List, Tuple, int], name=None,
             list_length=None):
    r"""Converts `hparams` value into a list.

    If :attr:`list_length` is given, then the canonicalized :attr:`value`
    must be of length :attr:`list_length`.
    """
    if not isinstance(value, (list, tuple)):
        if list_length is not None:
            value = [value] * list_length
        else:
            value = [value]
    if list_length is not None and len(value) != list_length:
        name = '' if name is None else name
        raise ValueError("hparams '%s' must be a list of length %d"
                         % (name, list_length))
    return value


class Conv1DNetwork(FeedForwardNetworkBase):
    r"""Simple `Conv-1D` network which consists of a sequence of convolutional
    layers followed with a sequence of dense layers.

    Args:
        in_channels (int): Number of channels in the input tensor.
        in_features (int): Size of the feature dimension in the input tensor.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    See :meth:`forward` for the inputs and outputs. If :attr:`"data_format"` is
    set to ``"channels_first"`` (this is the default), inputs must be a tensor
    of shape `[batch_size, channels, length]`. If :attr:`"data_format"` is set
    to ``"channels_last"``, inputs must be a tensor of shape
    `[batch_size, length, channels]`. For example, for sequence classification,
    `length` corresponds to time steps, and `channels` corresponds to embedding
    dim.

    Example:

    .. code-block:: python

        nn = Conv1DNetwork(in_channels=20, in_features=256) # Use the default

        inputs = torch.randn([64, 20, 256])
        outputs = nn(inputs)
        # outputs == Tensor of shape [64, 256], because the final dense layer
        # has size 256.

    .. document private functions
    """

    def __init__(self, in_channels: int, in_features: Optional[int] = None,
                 hparams=None):
        super().__init__(hparams=hparams)
        if self.hparams.num_dense_layers > 0 and in_features is None:
            raise ValueError("\"in_features\" cannot be None "
                             "if \"num_dense_layers\" > 0")

        # construct only non-dense layers first
        layer_hparams = self._build_non_dense_layer_hparams(
            in_channels=in_channels)
        self._build_layers(layers=None, layer_hparams=layer_hparams)
        if self.hparams.num_dense_layers > 0:
            if in_features is None:
                raise ValueError("\"in_features\" cannot be None "
                                 "if \"num_dense_layers\" > 0")
            ones = torch.ones(1, in_channels, in_features)
            input_size = self._infer_dense_layer_input_size(ones)
            layer_hparams = self._build_dense_hparams(
                in_features=input_size[1], layer_hparams=layer_hparams)
            self._build_layers(layers=None, layer_hparams=layer_hparams)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Conv layers
                "num_conv_layers": 1,
                "out_channels": 128,
                "kernel_size": [3, 4, 5],
                "conv_activation": "ReLU",
                "conv_activation_kwargs": None,
                "other_conv_kwargs": {},
                "data_format": "channels_first",
                # (2) Pooling layers
                "pooling": "MaxPool1d",
                "pool_size": None,
                "pool_stride": 1,
                "other_pool_kwargs": {},
                # (3) Dense layers
                "num_dense_layers": 1,
                "out_features": 256,
                "dense_activation": None,
                "dense_activation_kwargs": None,
                "final_dense_activation": None,
                "final_dense_activation_kwargs": None,
                "other_dense_kwargs": None,
                # (4) Dropout
                "dropout_conv": [1],
                "dropout_dense": [],
                "dropout_rate": 0.75,
                # (5) Others
                "name": "conv1d_network"
            }

        Here:

        1. For **convolutional** layers:

           `"num_conv_layers"`: int
               Number of convolutional layers.

           `"out_channels"`: int or list
               The number of out_channels in the convolution, i.e., the
               dimensionality of the output space.

               - If ``"num_conv_layers"`` > 1 and ``"out_channels"`` is an int,
                 all convolution layers will have the same number of output
                 channels.

               - If ``"num_conv_layers"`` > 1 and ``"out_channels"`` is a list,
                 the length must equal ``"num_conv_layers"``. The number of
                 output channels of each convolution layer will be the
                 corresponding element from this list.

           `"kernel_size"`: int or list
               Lengths of 1D convolution windows.

               - If `"num_conv_layers"` = 1, this can also be a ``int`` list of
                 arbitrary length denoting differently sized convolution
                 windows. The number of output channels of each size is
                 specified by ``"out_channels"``.
                 For example, the default values will create 3 convolution
                 layers, each of which has kernel size of 3, 4, and 5,
                 respectively, and has output channel 128.

               - If `"num_conv_layers"` > 1, this must be a list of length
                 ``"num_conv_layers"``. Each element can be an ``int`` or a
                 ``int`` list of arbitrary length denoting the kernel size of
                 each layer.

           `"conv_activation"`: str or callable
               Activation applied to the output of the convolutional
               layers. Set to `None` to maintain a linear activation.
               See :func:`~texar.torch.core.get_layer` for more details.

           `"conv_activation_kwargs"`: dict, optional
               Keyword arguments for the activation following the convolutional
               layer. See :func:`~texar.torch.core.get_layer` for more details.

           `"other_conv_kwargs"`: list or dict, optional
               Other keyword arguments for :torch_nn:`Conv1d` constructor,
               e.g., ``padding``.

               - If a dict, the same dict is applied to all the convolution
                 layers.

               - If a list, the length must equal ``"num_conv_layers"``. This
                 list can contain nested lists. If the convolution layer at
                 index i has multiple kernel sizes, then the corresponding
                 element of this list can also be a list of length equal to
                 ``"kernel_size"`` at index i. If the element at index i is
                 instead a dict, then the same dict gets applied to all the
                 convolution layers at index i.

           `"data_format"`: str, optional
               Data format of the input tensor. Defaults to ``channels_first``
               denoting the first dimension to be the channel dimension. Set it
               to ``channels_last`` to treat last dimension as the channel
               dimension. This argument can also be passed in ``forward``
               function, in which case the value specified here will be
               ignored.

        2. For **pooling** layers:

           `"pooling"`: str or class or instance
               Pooling layer after each of the convolutional layer(s). Can be a
               pooling layer class, its name or module path, or a class
               instance.

           `"pool_size"`: int or list, optional
               Size of the pooling window. If an ``int``, all pooling layer
               will have the same pool size. If a list, the list length must
               equal ``"num_conv_layers"``. If `None` and the pooling type
               is either :torch_docs:`MaxPool1d <nn.html#maxpool1d>` or
               :torch_docs:`AvgPool1d <nn.html#avgpool1d>`, the pool size will
               be set to input size. That is, the output of the pooling layer
               is a single unit.

           `"pool_stride"`: int or list, optional
               Strides of the pooling operation. If an ``int``, all
               layers will have the same stride. If a list, the list length
               must equal ``"num_conv_layers"``.

           `"other_pool_kwargs"`: list or dict, optional
               Other keyword arguments for pooling layer class constructor.

               - If a dict, the same dict is applied to all the pooling layers.

               - If a list, the length must equal ``"num_conv_layers"``. The
                 pooling arguments for layer i will be the element at index i
                 from this list.

        3. For **dense** layers (note that here dense layers always follow
           convolutional and pooling layers):

           `"num_dense_layers"`: int
               Number of dense layers.

           `"out_features"`: int or list
               Dimension of features after the dense layers. If an
               ``int``, all dense layers will have the same feature dimension.
               If a list of ``int``, the list length must equal
               ``"num_dense_layers"``.

           `"dense_activation"`: str or callable
               Activation function applied to the output of the dense
               layers **except** the last dense layer output. Set to
               `None` to maintain a linear activation.

           `"dense_activation_kwargs"`: dict, optional
               Keyword arguments for dense layer activation functions before
               the last dense layer.

           `"final_dense_activation"`: str or callable
               Activation function applied to the output of the **last** dense
               layer. Set to `None` to maintain a linear activation.

           `"final_dense_activation_kwargs"`: dict, optional
               Keyword arguments for the activation function of last
               dense layer.

           `"other_dense_kwargs"`: dict, optional
               Other keyword arguments for dense layer class constructor.

        4. For **dropouts**:

           `"dropout_conv"`: int or list
               The indices of convolutional layers (starting from 0) whose
               **inputs** are applied with dropout.
               The index = :attr:`num_conv_layers` means dropout applies to the
               final convolutional layer output. For example,

               .. code-block:: python

                   {
                       "num_conv_layers": 2,
                       "dropout_conv": [0, 2]
                   }

               will leads to a series of layers as
               `-dropout-conv0-conv1-dropout-`.

               The dropout mode (training or not) is controlled
               by :attr:`self.training`.

           `"dropout_dense"`: int or list
               Same as ``"dropout_conv"`` but applied to dense layers (index
               starting from 0).

           `"dropout_rate"`: float
               The dropout rate, between 0 and 1. For example,
               ``"dropout_rate": 0.1`` would drop out 10% of elements.

        5. Others:

           `"name"`: str
               Name of the network.
        """
        return {
            # (1) Conv layers
            "num_conv_layers": 1,
            "out_channels": 128,
            "kernel_size": [3, 4, 5],
            "conv_activation": "ReLU",
            "conv_activation_kwargs": None,
            "other_conv_kwargs": {},
            "data_format": "channels_first",
            # (2) Pooling layers
            "pooling": "MaxPool1d",
            "pool_size": None,
            "pool_stride": 1,
            "other_pool_kwargs": {},
            # (3) Dense layers
            "num_dense_layers": 1,
            "out_features": 256,
            "dense_activation": None,
            "dense_activation_kwargs": None,
            "final_dense_activation": None,
            "final_dense_activation_kwargs": None,
            "other_dense_kwargs": None,
            # (4) Dropout
            "dropout_conv": [1],
            "dropout_dense": [],
            "dropout_rate": 0.75,
            # (5) Others
            "name": "conv1d_network",
            "@no_typecheck": ["out_channels", "kernel_size", "conv_activation",
                              "other_conv_kwargs", "pool_size", "pool_stride",
                              "other_pool_kwargs", "out_features",
                              "dense_activation", "dropout_conv",
                              "dropout_dense"]
        }

    def _build_pool_hparams(self):
        pool_type = self._hparams.pooling
        if pool_type == "MaxPool":
            pool_type = "MaxPool1d"
        elif pool_type == "AvgPool":
            pool_type = "AvgPool1d"

        npool = self._hparams.num_conv_layers
        kernel_size = _to_list(self._hparams.pool_size, "pool_size", npool)
        stride = _to_list(self._hparams.pool_stride, "pool_stride", npool)

        other_kwargs = self._hparams.other_pool_kwargs
        if isinstance(other_kwargs, HParams):
            other_kwargs = other_kwargs.todict()
            other_kwargs = _to_list(other_kwargs, "other_kwargs", npool)
        elif isinstance(other_kwargs, (list, tuple)):
            if len(other_kwargs) != npool:
                raise ValueError("The length of hparams['other_pool_kwargs'] "
                                 "must equal 'num_conv_layers'")
        else:
            raise ValueError("hparams['other_pool_kwargs'] must be either a "
                             "dict or list/tuple")

        pool_hparams = []
        for i in range(npool):
            kwargs_i = {"kernel_size": kernel_size[i], "stride": stride[i]}
            kwargs_i.update(other_kwargs[i])
            pool_hparams_ = get_pooling_layer_hparams({"type": pool_type,
                                                       "kwargs": kwargs_i})
            pool_hparams.append(pool_hparams_)

        return pool_hparams

    def _build_conv1d_hparams(self, in_channels, pool_hparams):
        r"""Creates the hparams for each of the convolutional layers usable for
        :func:`texar.torch.core.layers.get_layer`.
        """
        nconv = self._hparams.num_conv_layers
        if len(pool_hparams) != nconv:
            raise ValueError("`pool_hparams` must be of length %d" % nconv)

        in_channels = [in_channels]
        out_channels = _to_list(self._hparams.out_channels, 'out_channels',
                                nconv)
        # because in_channels(i) = out_channels(i-1)
        in_channels.extend(out_channels[:-1])

        if nconv == 1:
            kernel_size = _to_list(self._hparams.kernel_size)
            if not isinstance(kernel_size[0], (list, tuple)):
                kernel_size = [kernel_size]
        elif nconv > 1:
            kernel_size = _to_list(self._hparams.kernel_size,
                                   'kernel_size', nconv)
            kernel_size = [_to_list(ks) for ks in kernel_size]

        other_kwargs = self._hparams.other_conv_kwargs
        if isinstance(other_kwargs, HParams):
            other_kwargs = other_kwargs.todict()
            other_kwargs = _to_list(other_kwargs, "other_conv_kwargs", nconv)
        elif isinstance(other_kwargs, (list, tuple)):
            if len(other_kwargs) != nconv:
                raise ValueError("The length of hparams['other_conv_kwargs'] "
                                 "must be equal to 'num_conv_layers'")
        else:
            raise ValueError("hparams['other_conv_kwargs'] must be a either "
                             "a dict or a list.")

        def _activation_hparams(name, kwargs=None):
            if kwargs is not None:
                return {"type": name, "kwargs": kwargs}
            else:
                return {"type": name, "kwargs": {}}

        conv_pool_hparams = []
        for i in range(nconv):
            hparams_i = []
            names = []
            if isinstance(other_kwargs[i], dict):
                other_kwargs[i] = _to_list(other_kwargs[i], "other_kwargs[i]",
                                           len(kernel_size[i]))
            elif (isinstance(other_kwargs[i], (list, tuple))
                  and len(other_kwargs[i]) != kernel_size[i]):
                raise ValueError("The length of hparams['other_conv_kwargs'][i]"
                                 " must be equal to the length of "
                                 "hparams['kernel_size'][i]")
            for idx, ks_ij in enumerate(kernel_size[i]):
                name = uniquify_str("conv_%d" % (i + 1), names)
                names.append(name)
                conv_kwargs_ij = {
                    "in_channels": in_channels[i],
                    "out_channels": out_channels[i],
                    "kernel_size": ks_ij
                }
                conv_kwargs_ij.update(other_kwargs[i][idx])
                hparams_i.append(
                    {"type": "Conv1d", "kwargs": conv_kwargs_ij})
            if len(hparams_i) == 1:
                if self._hparams.conv_activation:
                    layers = {
                        "layers": [hparams_i[0],
                                   _activation_hparams(
                                       self._hparams.conv_activation,
                                       self._hparams.conv_activation_kwargs)]}
                    sequential_layer = {"type": "Sequential", "kwargs": layers}
                    conv_pool_hparams.append([sequential_layer,
                                              pool_hparams[i]])
                else:
                    conv_pool_hparams.append([hparams_i[0], pool_hparams[i]])
            else:  # creates MergeLayer
                mrg_kwargs_layers = []
                for hparams_ij in hparams_i:
                    if self._hparams.conv_activation:
                        seq_kwargs_j = {
                            "layers": [
                                hparams_ij,
                                _activation_hparams(
                                    self._hparams.conv_activation,
                                    self._hparams.conv_activation_kwargs),
                                pool_hparams[i]
                            ]
                        }
                    else:
                        seq_kwargs_j = {"layers": [hparams_ij, pool_hparams[i]]}
                    mrg_kwargs_layers.append(
                        {"type": "Sequential", "kwargs": seq_kwargs_j})
                mrg_hparams = {"type": "MergeLayer",
                               "kwargs": {"layers": mrg_kwargs_layers}}
                conv_pool_hparams.append(mrg_hparams)

        return conv_pool_hparams

    def _build_dense_hparams(self, in_features: int, layer_hparams):
        ndense = self._hparams.num_dense_layers
        in_features = [in_features]
        out_features = _to_list(self._hparams.out_features, 'out_features',
                                ndense)
        # because in_features(i) = out_features(i-1)
        in_features.extend(out_features[:-1])
        other_kwargs = self._hparams.other_dense_kwargs or {}
        if isinstance(other_kwargs, HParams):
            other_kwargs = other_kwargs.todict()
        if not isinstance(other_kwargs, dict):
            raise ValueError("hparams['other_dense_kwargs'] must be a dict.")

        def _activation_hparams(name, kwargs=None):
            if kwargs is not None:
                return {"type": name, "kwargs": kwargs}
            else:
                return {"type": name, "kwargs": {}}

        dense_hparams = []
        for i in range(ndense):
            kwargs_i = {"in_features": in_features[i],
                        "out_features": out_features[i]}
            kwargs_i.update(other_kwargs)

            dense_hparams_i = {"type": "Linear", "kwargs": kwargs_i}
            if i < ndense - 1 and self._hparams.dense_activation is not None:
                layers = {
                    "layers": [dense_hparams_i,
                               _activation_hparams(
                                   self._hparams.dense_activation,
                                   self._hparams.dense_activation_kwargs)
                               ]}
                sequential_layer = {"type": "Sequential", "kwargs": layers}
                dense_hparams.append(sequential_layer)

            elif (i == ndense - 1 and
                  self._hparams.final_dense_activation is not None):
                layers = {
                    "layers": [dense_hparams_i,
                               _activation_hparams(
                                   self._hparams.final_dense_activation,
                                   self._hparams.final_dense_activation_kwargs)
                               ]}
                sequential_layer = {"type": "Sequential", "kwargs": layers}
                dense_hparams.append(sequential_layer)
            else:
                dense_hparams.append(dense_hparams_i)

        def _dropout_hparams():
            return {"type": "Dropout",
                    "kwargs": {"p": self._hparams.dropout_rate}}

        dropout_dense = _to_list(self._hparams.dropout_dense)
        ndense = self._hparams.num_dense_layers
        if ndense > 0:  # Add flatten layers before dense layers
            layer_hparams.append({"type": "Flatten"})
        for dense_i in range(ndense):
            if dense_i in dropout_dense:
                layer_hparams.append(_dropout_hparams())
            layer_hparams.append(dense_hparams[dense_i])
        if ndense in dropout_dense:
            layer_hparams.append(_dropout_hparams())

        return layer_hparams

    def _build_non_dense_layer_hparams(self, in_channels):
        pool_hparams = self._build_pool_hparams()
        conv_pool_hparams = self._build_conv1d_hparams(in_channels,
                                                       pool_hparams)

        def _dropout_hparams():
            return {"type": "Dropout",
                    "kwargs": {"p": self._hparams.dropout_rate}}

        dropout_conv = _to_list(self._hparams.dropout_conv)

        layers_hparams = []
        nconv = self._hparams.num_conv_layers
        for conv_i in range(nconv):
            if conv_i in dropout_conv:
                layers_hparams.append(_dropout_hparams())
            if isinstance(conv_pool_hparams[conv_i], (list, tuple)):
                layers_hparams += conv_pool_hparams[conv_i]
            else:
                layers_hparams.append(conv_pool_hparams[conv_i])
        if nconv in dropout_conv:
            layers_hparams.append(_dropout_hparams())

        return layers_hparams

    def forward(self,  # type: ignore
                input: torch.Tensor,
                sequence_length: Union[torch.LongTensor, List[int]] = None,
                dtype: Optional[torch.dtype] = None,
                data_format: Optional[str] = None) -> torch.Tensor:
        r"""Feeds forward inputs through the network layers and returns outputs.

        Args:
            input: The inputs to the network, which is a 3D tensor.
            sequence_length (optional): An :tensor:`LongTensor` of shape
                ``[batch_size]`` or a python array containing the length of
                each element in :attr:`inputs`. If given, time steps beyond
                the length will first be masked out before feeding to the
                layers.
            dtype (optional): Type of the inputs. If not provided,
                infers from inputs automatically.
            data_format (optional): Data type of the input tensor. If
                ``channels_last``, the last dimension will be treated as channel
                dimension so the size of the :attr:`input` should be
                `[batch_size, X, channel]`. If ``channels_first``, first
                dimension will be treated as channel dimension so the size
                should be `[batch_size, channel, X]`. Defaults to None.
                If None, the value will be picked from hyperparameters.
        Returns:
            The output of the final layer.
        """
        if input.dim() != 3:
            raise ValueError("'input' should be a 3D tensor.")

        if data_format is None:
            data_format = self.hparams["data_format"]

        if data_format == "channels_first":
            # masking requires channels in last dimension
            input = input.permute(0, 2, 1)

            if sequence_length is not None:
                input = mask_sequences(input, sequence_length,
                                       dtype=dtype, time_major=False)

            # network is constructed for channel first tensors
            input = input.permute(0, 2, 1)

            output = super().forward(input)

        elif data_format == "channels_last":
            if sequence_length is not None:
                input = mask_sequences(input, sequence_length,
                                       dtype=dtype, time_major=False)

            input = input.permute(0, 2, 1)

            output = super().forward(input)

            # transpose only when tensors are 3D
            if output.dim() == 3:
                output = output.permute(0, 2, 1)
        else:
            raise ValueError("Invalid 'data_format'")

        return output

    def _infer_dense_layer_input_size(self, input: torch.Tensor) -> torch.Size:
        # feed forward the input on the conv part of the network to infer
        # input shape for dense layers
        with torch.no_grad():
            output = super().forward(input)
        return output.view(output.size()[0], -1).size()

    @property
    def output_size(self) -> int:
        r"""The feature size of :meth:`forward` output.
        """
        if self.hparams.num_dense_layers <= 0:
            out_channels = self._hparams.out_channels
            if not isinstance(out_channels, (list, tuple)):
                out_channels = [out_channels]
            nconv = self._hparams.num_conv_layers
            if nconv == 1:
                kernel_size = _to_list(self._hparams.kernel_size)
                if not isinstance(kernel_size[0], (list, tuple)):
                    kernel_size = [kernel_size]
            elif nconv > 1:
                kernel_size = _to_list(self._hparams.kernel_size,
                                       'kernel_size', nconv)
                kernel_size = [_to_list(ks) for ks in kernel_size]
            return out_channels[-1] * len(kernel_size[-1])
        else:
            out_features = self._hparams.out_features
            if isinstance(out_features, (list, tuple)):
                return out_features[-1]
            else:
                return out_features
