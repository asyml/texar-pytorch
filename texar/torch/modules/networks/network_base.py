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
Base class for feed forward neural networks.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from texar.torch.core.layers import get_layer
from texar.torch.hyperparams import HParams
from texar.torch.module_base import ModuleBase
from texar.torch.utils.utils import uniquify_str

__all__ = [
    "FeedForwardNetworkBase",
]


class FeedForwardNetworkBase(ModuleBase):
    r"""Base class inherited by all feed-forward network classes.

    Args:
        hparams (dict, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    See :meth:`forward` for the inputs and outputs.
    """

    def __init__(self,
                 hparams: Optional[Union[HParams, Dict[str, Any]]] = None):
        super().__init__(hparams)

        self._layers = nn.ModuleList()
        self._layer_names: List[str] = []
        self._layers_by_name: Dict[str, nn.Module] = {}
        self._layer_outputs: List[torch.Tensor] = []
        self._layer_outputs_by_name: Dict[str, torch.Tensor] = {}

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "NN"
            }
        """
        return {
            "name": "NN"
        }

    def __repr__(self) -> str:
        if len(list(self.modules())) == 1:  # only contains `_layers`
            return ModuleBase.__repr__(self._layers)
        return super().__repr__()

    def forward(self,  # type: ignore
                input: torch.Tensor) -> torch.Tensor:
        r"""Feeds forward inputs through the network layers and returns outputs.

        Args:
            input: The inputs to the network. The requirements on inputs
                depends on the first layer and subsequent layers in the
                network.

        Returns:
            The output of the network.
        """
        outputs = input
        for layer in self._layers:
            outputs = layer(outputs)

        return outputs

    def append_layer(self, layer: Union[nn.Module, HParams, Dict[str, Any]]):
        r"""Appends a layer to the end of the network.

        Args:
            layer: A subclass of :torch_nn:`Module`, or a dict of layer
                hyperparameters.
        """
        layer_ = layer
        if not isinstance(layer_, nn.Module):
            layer_ = get_layer(hparams=layer_)
        self._layers.append(layer_)
        layer_name = uniquify_str(layer_.__class__.__name__, self._layer_names)
        self._layer_names.append(layer_name)
        self._layers_by_name[layer_name] = layer_

    def has_layer(self, layer_name: str) -> bool:
        r"""Returns `True` if the network with the name exists. Returns
        `False` otherwise.

        Args:
            layer_name (str): Name of the layer.
        """
        return layer_name in self._layers_by_name

    def layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        r"""Returns the layer with the name. Returns `None` if the layer name
        does not exist.

        Args:
            layer_name (str): Name of the layer.
        """
        return self._layers_by_name.get(layer_name, None)

    @property
    def layers_by_name(self) -> Dict[str, nn.Module]:
        r"""A dictionary mapping layer names to the layers.
        """
        return self._layers_by_name

    @property
    def layers(self) -> nn.ModuleList:
        r"""A list of the layers.
        """
        return self._layers

    @property
    def layer_names(self) -> List[str]:
        r"""A list of uniquified layer names.
        """
        return self._layer_names

    def _build_layers(self,
                      layers: Optional[nn.ModuleList] = None,
                      layer_hparams: Optional[List[
                          Union[HParams, Dict[str, Any]]]] = None):
        r"""Builds layers.

        Either :attr:`layer_hparams` or :attr:`layers` must be
        provided. If both are given, :attr:`layers` will be used.

        Args:
            layers (optional): A list of layer instances supplied as an instance
                of :torch_nn:`ModuleList`.
            layer_hparams (optional): A list of layer hparams, each to which
                is fed to :func:`~texar.torch.core.layers.get_layer` to create
                the layer instance.
        """
        if layers is not None:
            self._layers = layers
        else:
            if layer_hparams is None:
                raise ValueError(
                    'Either `layer` or `layer_hparams` is required.')
            self._layers = nn.ModuleList()
            for _, hparams in enumerate(layer_hparams):
                self._layers.append(get_layer(hparams=hparams))

        for layer in self._layers:
            layer_name = uniquify_str(layer.__class__.__name__,
                                      self._layer_names)
            self._layer_names.append(layer_name)
            self._layers_by_name[layer_name] = layer
