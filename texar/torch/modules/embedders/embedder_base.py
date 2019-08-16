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
from typing import Optional, Tuple

import torch
from torch import nn

from texar.torch.module_base import ModuleBase
from texar.torch.modules.embedders import embedder_utils

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
            specified in ``hparams["initializer"]``.
        hparams (dict or HParams, optional): Embedder hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.
    """

    def __init__(self, num_embeds: Optional[int] = None,
                 init_value: Optional[torch.Tensor] = None, hparams=None):
        super().__init__(hparams=hparams)

        if num_embeds is not None or init_value is not None:
            self._embedding = nn.Parameter(embedder_utils.get_embedding(
                num_embeds, init_value, hparams))

            self._num_embeds = self._embedding.size(0)

            self._dim_rank = self._embedding.dim() - 1
            if self._dim_rank == 1:
                self._dim = self._embedding.size(1)
            else:
                self._dim = self._embedding.size()[1:]  # type: ignore

    def _get_noise_shape(self, dropout_strategy: str,
                         ids_rank: Optional[int] = None,
                         dropout_input: Optional[torch.Tensor] = None) \
            -> Optional[Tuple[int, ...]]:

        if dropout_strategy == 'element':
            noise_shape = None
        elif dropout_strategy == 'item':
            assert dropout_input is not None
            assert ids_rank is not None
            shape_a = dropout_input.size()[:ids_rank]
            shape_b = (1,) * self._dim_rank
            noise_shape = shape_a + shape_b
        elif dropout_strategy == 'item_type':
            noise_shape = (self._num_embeds,) + (1,) * self._dim_rank
        else:
            raise ValueError(f"Unknown dropout strategy: {dropout_strategy}")
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

    @property
    def num_embeds(self) -> int:
        r"""The number of embedding elements.
        """
        return self._num_embeds


class EmbeddingDropout(ModuleBase):
    r"""The dropout layer that used for the embedding.

    Args:
        rate (float, required): The dropout rate applied to the embedding.
            For example, if rate is 0.1, 10% of the embedding will be zeroed
            out. Set to 0 to disable dropout.

        hparams (dict or HParams, optional): Embedder hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.
    """

    def __init__(self, rate: float, hparams=None):
        super().__init__(hparams=hparams)
        self._rate = rate

    def forward(self,  # type: ignore
                input_tensor: torch.Tensor,
                noise_shape: Optional[torch.Size] = None) -> torch.Tensor:
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
            noise_shape = input_tensor.size()
        keep_rate = 1 - self._rate
        mask = input_tensor.new_full(noise_shape, keep_rate)
        mask += input_tensor.new_empty(noise_shape).uniform_(0, 1)
        mask = torch.floor(mask).div_(keep_rate)
        return input_tensor * mask

    @property
    def output_size(self):
        raise ValueError("'output_size' can not be calculated "
                         "because it is equal to the input size.")
