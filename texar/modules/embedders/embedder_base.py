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
from torch.nn.parameter import Parameter

import texar.modules.embedders.embedder_utils as embedder_utils
from texar.module_base import ModuleBase

# pylint: disable=invalid-name

__all__ = [
    "EmbedderBase",
    "EmbeddingDropout",
]


class EmbedderBase(ModuleBase):
    r"""The base embedder class that all embedder classes inherit.

    Args:
        num_embeds (int, optional): The number of embedding elements, e.g.,
            the vocabulary size of a word embedder.
        init_value (Tensor or numpy array, optional): Initial values of the
            embedding variable. If not given, embedding is initialized as
            specified in :attr:`hparams["initializer"]`.
        hparams (dict or HParams, optional): Embedder hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self, num_embeds=None, init_value=None, hparams=None):
        ModuleBase.__init__(self, hparams)

        if num_embeds is not None or init_value is not None:
            self._embedding = Parameter(embedder_utils.get_embedding(
                num_embeds, init_value, hparams))

            self._num_embeds = self._embedding.shape[0]

            self._dim = self._embedding.shape[1:]
            self._dim_rank = len(self._dim)
            if self._dim_rank == 1:
                self._dim = self._dim[0]

    def _get_noise_shape(self, hparams, ids_rank=None, dropout_input=None,
                         dropout_strategy=None):

        st = dropout_strategy
        st = hparams.dropout_strategy if st is None else st
        if st == 'element':
            noise_shape = None
        elif st == 'item':
            shape_a = list(dropout_input.shape[:ids_rank])
            shape_b = [1] * self._dim_rank
            noise_shape = shape_a + shape_b
        elif st == 'item_type':
            noise_shape = [self._num_embeds] + [1] * self._dim_rank
        else:
            raise ValueError('Unknown dropout strategy: {}'.format(st))
        return noise_shape

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

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
        r"""The number of embedding elements.
        """
        return self._num_embeds


class EmbeddingDropout(ModuleBase):
    r"""The dropout layer that used for the embedding.

    Args:
        rate (float, required): The dropout rate applied to the embedding.
            E.g., if rate is 0.1, 10% of the embedding will be dropped out.
            Set to 0 to disable dropout.

        hparams (dict or HParams, optional): Embedder hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self, rate=None, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._rate = rate

    # pylint: disable=W0221
    def forward(self, input_tensor=None, noise_shape=None):
        r"""Apply dropout on the tensor.

        Args:
            input_tensor: The tensor to apply dropout on.
            noise_shape (list, optional): The shape of the noise mask which
                specifies the dropout dimensions for the embedding.

        Returns:
            The tensor after applying dropout.
        """
        if not self.training or self._rate == 0.0:
            return input_tensor
        if noise_shape is None:
            noise_shape = input_tensor.shape
        keep_rate = 1 - self._rate
        mask = input_tensor.new_full(noise_shape, keep_rate)
        mask += input_tensor.new_empty(noise_shape).uniform_(0, 1)
        mask = torch.floor(mask).div_(keep_rate)
        return input_tensor * mask
