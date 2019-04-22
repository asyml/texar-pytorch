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
The base embedder class.
"""

import torch

from texar.module_base import ModuleBase
from texar.modules.embedders import embedder_utils

# pylint: disable=invalid-name

__all__ = [
    "EmbedderBase"
]

class EmbedderBase(ModuleBase):
    """The base embedder class that all embedder classes inherit.

    Args:
        num_embeds (int, optional): The number of embedding elements, e.g.,
            the vocabulary size of a word embedder.
        hparams (dict or HParams, optional): Embedder hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self, num_embeds=None, hparams=None):
        ModuleBase.__init__(self, hparams)

        self._num_embeds = num_embeds

    # pylint: disable=attribute-defined-outside-init
    def _init_parameterized_embedding(self, init_value, num_embeds, hparams):
        self._embedding = embedder_utils.get_embedding(
            hparams, init_value, num_embeds)

        self._num_embeds = list(self._embedding.shape)[0]

        self._dim = list(self._embedding.shape)[1:]
        self._dim_rank = len(self._dim)
        if self._dim_rank == 1:
            self._dim = self._dim[0]

    def _get_dropout_layer(self, hparams, ids_rank=None, dropout_input=None,
                           dropout_strategy=None):
        """Creates dropout layer according to dropout strategy.

        Called in :meth:`_build()`.
        """
        dropout_layer = None

        st = dropout_strategy
        st = hparams.dropout_strategy if st is None else st
        noise_shape = None

        if hparams.dropout_rate > 0.:
            if st == 'element':
                noise_shape = None
            elif st == 'item':
                # pylint: disable=not-callable
                shape_a = torch.tensor(
                    dropout_input.shape[:ids_rank]).type(torch.int32)
                shape_b = torch.ones([self._dim_rank], dtype=torch.int32)
                noise_shape = torch.cat(
                    (shape_a, shape_b), dim=0).tolist()
            elif st == 'item_type':
                noise_shape = [self._num_embeds] + [1] * self._dim_rank
            else:
                raise ValueError('Unknown dropout strategy: {}'.format(st))

            dropout_layer = EmbeddingDropout(
                rate=hparams.dropout_rate,
                noise_shape=noise_shape)

        return dropout_layer

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "embedder"
            }
        """
        return {
            "name": "embedder"
        }

    # pylint: disable=W0221
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def num_embeds(self):
        """The number of embedding elements.
        """
        return self._num_embeds

class EmbeddingDropout(ModuleBase):
    """The dropout layer that used for the embedding.

    Args:
        rate (int, required): The dropout rate applied to the embedding.

        noise_shape (list, optional): The shape of the noise mask which
            can specified the dropout dimensions for the embedding.

        hparams (dict or HParams, optional): Embedder hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """
    def __init__(self, rate=None, noise_shape=None, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._rate = rate
        self._noise_shape = noise_shape
    # pylint: disable=W0221
    def forward(self, input_tensor):
        if not self.training or self._rate == 0:
            return input_tensor
        if self._noise_shape is None:
            self._noise_shape = input_tensor.shape
        keep_rate = 1 - self._rate
        mask = torch.empty(self._noise_shape).fill_(keep_rate)
        mask += torch.empty(self._noise_shape).uniform_(0, 1)
        mask = torch.floor(mask).div_(keep_rate)
        return input_tensor * mask
    