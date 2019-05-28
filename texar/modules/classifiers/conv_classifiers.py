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
Various classifier classes.
"""
# pylint: disable=not-context-manager, too-many-arguments, too-many-locals

import torch
from torch import nn

from typing import Optional, List, Dict, Tuple, Union, Any

from texar.modules.classifiers.classifier_base import ClassifierBase
from texar.modules.encoders.conv_encoders import Conv1DEncoder
from texar.utils import utils
from texar.hyperparams import HParams

__all__ = [
    "Conv1DClassifier"
]


class Conv1DClassifier(ClassifierBase):
    """Simple Conv-1D classifier.
    This is a combination of the
    :class:`~texar.modules.Conv1DEncoder` with a classification layer.

    Args:
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    Example:

        .. code-block:: python

            clas = Conv1DClassifier(hparams={'num_classes': 10})

            inputs = tf.random_uniform([64, 20, 256])
            logits, pred = clas(inputs)
            # logits == Tensor of shape [64, 10]
            # pred   == Tensor of shape [64]

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, in_channels: int, in_features: Optional[int] = None,
                 hparams: Optional[Union[HParams, Dict[str, Any]]] = None):
        ClassifierBase.__init__(self, hparams)

        encoder_hparams = utils.dict_fetch(hparams,
                                           Conv1DEncoder.default_hparams())
        self._encoder = Conv1DEncoder(in_channels=in_channels,
                                      in_features=in_features,
                                      hparams=encoder_hparams)

        # Add an additional dense layer if needed
        self._num_classes = self._hparams.num_classes
        if self._num_classes > 0:
            if self._hparams.num_dense_layers <= 0:
                self._encoder.append_layer({"type": "Flatten"})

            logit_kwargs = self._hparams.logit_layer_kwargs
            if logit_kwargs is None:
                logit_kwargs = {}
            elif not isinstance(logit_kwargs, HParams):
                raise ValueError(
                    "hparams['logit_layer_kwargs'] must be a dict.")
            else:
                logit_kwargs = logit_kwargs.todict()
            logit_kwargs.update({"out_features": self._num_classes})

            self._encoder.append_layer({"type": "Linear",
                                        "kwargs": logit_kwargs})

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Same hyperparameters as in Conv1DEncoder
                ...

                # (2) Additional hyperparameters
                "num_classes": 2,
                "logit_layer_kwargs": {
                    "use_bias": False
                },
                "name": "conv1d_classifier"
            }

        Here:

        1. Same hyperparameters as in :class:`~texar.modules.Conv1DEncoder`.
        See the :meth:`~texar.modules.Conv1DEncoder.default_hparams`.
        An instance of Conv1DEncoder is created for feature extraction.

        2. Additional hyperparameters:

            "num_classes" : int
                Number of classes:

                - If **`> 0`**, an additional :tf_main:`Dense <layers/Dense>` \
                layer is appended to the encoder to compute the logits over \
                classes.
                - If **`<= 0`**, no dense layer is appended. The number of \
                classes is assumed to be equal to "out_features" of the final
                dense layer size of the encoder.

            "logit_layer_kwargs" : dict
                Keyword arguments for the logit Dense layer constructor,
                except for argument "out_features" which is set to
                "num_classes". Ignored if no extra logit layer is appended.

            "name" : str
                Name of the classifier.
        """
        hparams = Conv1DEncoder.default_hparams()
        hparams.update({
            "name": "conv1d_classifier",
            "num_classes": 2,  # set to <=0 to avoid appending output layer
            "logit_layer_kwargs": {
                "in_features": hparams["out_features"],
                "bias": True
            }
        })
        return hparams

    def forward(self,  # type:ignore
                input: torch.Tensor,
                sequence_length: Union[torch.LongTensor, List[int]] = None,
                dtype: Optional[torch.dtype] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Feeds the inputs through the network and makes classification.

        The arguments are the same as in :class:`~texar.modules.Conv1DEncoder`.

        The predictions of binary classification ("num_classes"=1) and
        multi-way classification ("num_classes">1) are different, as explained
        below.

        Args:
            input: The inputs to the network, which is a 3D tensor. See
                :class:`~texar.modules.Conv1DEncoder` for more details.
            sequence_length (optional): An int tensor of shape `[batch_size]` or
                a python array containing the length of each element in
                :attr:`inputs`. If given, time steps beyond the length will
                first be masked out before feeding to the layers.
            dtype (optional): Type of the inputs. If not provided, infers
                from inputs automatically.

        Returns:
            A tuple `(logits, pred)`, where

            - **`logits`** is a Tensor of shape `[batch_size, num_classes]`\
            for `num_classes` >1, and `[batch_size]` for `num_classes` =1 \
            (i.e., binary classification).
            - **`pred`** is the prediction, a Tensor of shape `[batch_size]` \
            and type `tf.int64`. For binary classification, the standard \
            sigmoid function is used for prediction, and the class labels are \
            `{0, 1}`.
        """
        logits = self._encoder(input, sequence_length=sequence_length,
                               dtype=dtype)

        num_classes = self._hparams.num_classes
        is_binary = num_classes == 1
        is_binary = is_binary or (num_classes <= 0 and logits.shape[1] == 1)

        if is_binary:
            pred = (logits > 0)
            logits = logits.view(-1)
        else:
            pred = torch.argmax(logits, dim=1)

        pred = pred.view(-1).long()

        return logits, pred

    @property
    def num_classes(self) -> int:
        """The number of classes.
        """
        return self._num_classes

    @property
    def encoder(self) -> nn.Module:
        """The classifier neural network.
        """
        return self._encoder

    def has_layer(self, layer_name) -> bool:
        """Returns `True` if the network with the name exists. Returns `False`
        otherwise.

        Args:
            layer_name (str): Name of the layer.
        """
        return self._encoder.has_layer(layer_name)

    def layer_by_name(self, layer_name) -> Optional[nn.Module]:
        """Returns the layer with the name. Returns 'None' if the layer name
        does not exist.

        Args:
            layer_name (str): Name of the layer.
        """
        return self._encoder.layer_by_name(layer_name)

    @property
    def layers_by_name(self) -> Dict[str, nn.Module]:
        """A dictionary mapping layer names to the layers.
        """
        return self._encoder.layers_by_name

    @property
    def layers(self) -> nn.ModuleList:
        """A list of the layers.
        """
        return self._encoder.layers

    @property
    def layer_names(self) -> List[str]:
        """A list of uniquified layer names.
        """
        return self._encoder.layer_names

