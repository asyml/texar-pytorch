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
Transformer encoders with multi-head self attention.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from mypy_extensions import TypedDict

from texar.torch.core import layers
from texar.torch.modules.encoders.encoder_base import EncoderBase
from texar.torch.utils.types import MaybeList

__all__ = [
    'MultiheadAttentionEncoder',
    'Cache',
]


class LayerCache(TypedDict):
    r"""Cache (state) for a single self-attention layer in
    :class:`MultiheadAttentionEncoder`.
    """
    keys: MaybeList[torch.Tensor]
    values: MaybeList[torch.Tensor]


class Cache(TypedDict):
    r"""Cache (state) for the entire :class:`MultiheadAttentionEncoder`.
    """
    memory: Optional[torch.Tensor]
    memory_attention_bias: Optional[torch.Tensor]
    layers: List[LayerCache]


class MultiheadAttentionEncoder(EncoderBase):
    r"""Multi-head Attention Encoder.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    .. document private functions
    """

    def __init__(self, input_size: int, hparams=None):
        super().__init__(hparams=hparams)
        use_bias = self._hparams.use_bias

        self.Q_dense = nn.Linear(input_size, self._hparams.num_units,
                                 bias=use_bias)
        self.K_dense = nn.Linear(input_size, self._hparams.num_units,
                                 bias=use_bias)
        self.V_dense = nn.Linear(input_size, self._hparams.num_units,
                                 bias=use_bias)
        self.O_dense = nn.Linear(self._hparams.num_units,
                                 self._hparams.output_dim, bias=use_bias)

        if self._hparams.initializer:
            # TODO(haoransh): we may define kernel_initializer and bias
            #  initializer seperately
            initialize = layers.get_initializer(self._hparams.initializer)
            assert initialize is not None
            for name, param in self.named_parameters():
                if name.split('.')[-1] == 'weight':
                    print('name:{}'.format(name))
                    initialize(param)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "initializer": None,
                'num_heads': 8,
                'output_dim': 512,
                'num_units': 512,
                'dropout_rate': 0.1,
                'use_bias': False,
                "name": "multihead_attention"
            }

        Here:

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.torch.core.get_initializer` for details.

        `"num_heads"`: int
            Number of heads for attention calculation.

        `"output_dim"`: int
            Output dimension of the returned tensor.

        `"num_units"`: int
            Hidden dimension of the unsplit attention space.
            Should be divisible by `"num_heads"`.

        `"dropout_rate"`: float
            Dropout rate in the attention.

        `"use_bias"`: bool
            Use bias when projecting the key, value and query.

        `"name"`: str
            Name of the module.
        """
        return {
            'initializer': None,
            'num_heads': 8,
            'output_dim': 512,
            'num_units': 512,
            'dropout_rate': 0.1,
            'use_bias': False,
            'name': 'multihead_attention',
        }

    def forward(self,  # type: ignore
                queries: torch.Tensor,
                memory: torch.Tensor,
                memory_attention_bias: torch.Tensor,
                cache: Optional[LayerCache] = None) \
            -> torch.Tensor:
        r"""Encodes the inputs.

        Args:
            queries: A 3D tensor with shape of
                ``[batch, length_query, depth_query]``.
            memory: A 3D tensor with shape of
                ``[batch, length_key, depth_key]``.
            memory_attention_bias: A 3D tensor with shape of
                ``[batch, length_key, num_units]``.
            cache: Memory cache only when inferring the sentence from scratch.

        Returns:
            A tensor of shape ``[batch_size, max_time, dim]`` containing the
            encoded vectors.
        """

        num_heads = self._hparams.num_heads
        num_units = self._hparams.num_units
        if num_units % num_heads != 0:
            raise ValueError(
                f"Value depth ({num_units}) must be divisible by "
                f"the number of attention heads ({num_heads}).")

        def _update_and_return(layer: nn.Module, key: str):
            if memory is None:
                # Self Attention
                out = layer(queries)

                if cache is not None:
                    # decoder self attention when dynamic decoding
                    res: MaybeList[torch.Tensor] = cache[key]
                    if isinstance(res, list):
                        # inference-like decoding
                        res.append(out.squeeze(1))
                        out = torch.stack(res, dim=1)
                    else:
                        # normal decoding
                        res = torch.cat([res, out], dim=1)
                        out = res
                    cache[key] = res

            else:
                # encoder decoder attention
                if cache is not None:
                    res: MaybeList[torch.Tensor] = cache[key]  # type: ignore
                    if isinstance(res, list):
                        # inference-like decoding
                        if len(res) == 0:
                            out = layer(memory)
                        else:
                            out = torch.stack(res, dim=1)
                    else:
                        # normal decoding
                        if res.size(1) == 0:
                            out = layer(memory)
                        else:
                            out = res
                else:
                    out = layer(memory)

            return out

        Q = self.Q_dense(queries)
        K = _update_and_return(self.K_dense, 'keys')
        V = _update_and_return(self.V_dense, 'values')

        Q_ = self._split_heads(Q)
        K_ = self._split_heads(K)
        V_ = self._split_heads(V)
        # [batch_size, num_heads, seq_length, memory_depth]
        key_depth_per_head = num_units // num_heads
        Q_ *= key_depth_per_head ** -0.5

        logits = torch.matmul(Q_, K_.transpose(-2, -1))
        if memory_attention_bias is not None:
            memory_attention_bias = memory_attention_bias.to(
                device=logits.device)
            logits += memory_attention_bias
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, self._hparams.dropout_rate, self.training)
        outputs = torch.matmul(weights, V_)

        outputs = self._combine_heads(outputs)
        outputs = self.O_dense(outputs)
        # (batch_size, length_query, output_dim)

        return outputs

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        r"""Split channels (dimension 2) into multiple heads,
        becomes dimension 1). Must ensure ``x.shape[-1]`` can be
        divided by num_heads.
        """
        depth = x.size(-1)
        split_x = torch.reshape(x, (
            x.size(0), x.size(1),
            self._hparams.num_heads, depth // self._hparams.num_heads))
        return split_x.permute((0, 2, 1, 3))

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x: A Tensor of shape ``[batch, num_heads, seq_len, dim]``
        Returns:
            A Tensor of shape ``[batch, seq_len, num_heads * dim]``
        """
        t = x.permute((0, 2, 1, 3))  # [batch, seq_len, num_heads, dim]
        num_heads, dim = t.size()[-2:]
        assert num_heads == self._hparams.num_heads
        return torch.reshape(t, (t.size(0), t.size(1), num_heads * dim))

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output.
        """
        return self._hparams.output_dim
