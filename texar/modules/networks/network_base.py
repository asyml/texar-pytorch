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

import torch

from texar.module_base import ModuleBase
from texar.utils import TexarError
from texar.core.layers import get_layer
from texar.utils.utils import uniquify_str
from texar.utils.mode import is_train_mode

# pylint: disable=too-many-instance-attributes, arguments-differ
# pylint: disable=protected-access

__all__ = [
    "_build_layers",
    "FeedForwardNetworkBase"
]


def _build_layers(network, layers=None, layer_hparams=None):
    """Builds layers.

    Either :attr:`layer_hparams` or :attr:`layers` must be
    provided. If both are given, :attr:`layers` will be used.

    Args:
        network: An instance of a subclass of
            :class:`~texar.modules.networks.network_base.FeedForwardNetworkBase`
        layers (optional): A list of layer instances.
        layer_hparams (optional): A list of layer hparams, each to which
            is fed to :func:`~texar.core.layers.get_layer` to create the
            layer instance.
    """
    if layers is not None:
        network._layers = layers
    else:
        if layer_hparams is None:
            raise ValueError(
                'Either `layer` or `layer_hparams` is required.')
        network._layers = []
        for _, hparams in enumerate(layer_hparams):
            network._layers.append(get_layer(hparams=hparams))

    for layer in network._layers:
        layer_name = uniquify_str(layer.name, network._layer_names)
        network._layer_names.append(layer_name)
        network._layers_by_name[layer_name] = layer


class FeedForwardNetworkBase(ModuleBase):
    """Base class inherited by all feed-forward network classes.

        Args:
            hparams (dict, optional): Hyperparameters. Missing
                hyperparamerter will be set to default values. See
                :meth:`default_hparams` for the hyperparameter sturcture and
                default values.

        See :meth:`_build` for the inputs and outputs.
        """

    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

        self._layers = []
        self._layer_names = []
        self._layers_by_name = {}
        self._layer_outputs = []
        self._layer_outputs_by_name = {}
