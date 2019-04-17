# Copyright 2018 The Texar Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import tensorflow as tf
import torch
import numpy as np
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
        '''if hparams.trainable:
            self._add_trainable_variable(self._embedding)
        '''
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
                noise_shape = np.concatenate((np.array(dropout_input.shape[:ids_rank]), np.ones([self._dim_rank], dtype=np.int32)), axis=0).tolist()
            elif st == 'item_type':
                noise_shape = [dropout_input.shape[0]] + [1] * self._dim_rank
                #noise_shape = [1] + [1] * self._dim_rank
            else:
                raise ValueError('Unknown dropout strategy: {}'.format(st))

            '''dropout_layer = tf.layers.Dropout(
                rate=hparams.dropout_rate, noise_shape=noise_shape)'''
            #print("hparams.dropout_rate", hparams.dropout_rate)
            if noise_shape is not None:
                dropout_layer = Dropout_with_mask(rate=hparams.dropout_rate, noise_shape=noise_shape)
            else:
                dropout_layer = torch.nn.Dropout(p=hparams.dropout_rate)

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def num_embeds(self):
        """The number of embedding elements.
        """
        return self._num_embeds

class Dropout_with_mask(ModuleBase):
    def __init__(self, rate=None, noise_shape=None, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._rate = rate
        self._noise_shape=noise_shape
        if noise_shape is not None:
            keep_rate = 1 - rate
            mask = torch.empty(noise_shape).fill_(keep_rate)
            mask += torch.empty(noise_shape).uniform_(0, 1)
            mask = torch.floor(mask).div_(keep_rate)
            self._mask = mask

    def forward(self, input_tensor):
        if self._noise_shape is not None:
            print("=" * 80)
            print("self._noise_shape", self._noise_shape)
            print("self._mask.shape", self._mask.shape)
            print("input_tensor.shape", input_tensor.shape)
            print(self._mask)
            return input_tensor * self._mask