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

import ast
import math
from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F
from torch import nn

from texar.torch.core import layers
from texar.torch.module_base import ModuleBase
from texar.torch.modules.encoders.multihead_attention import LayerCache
from texar.torch.utils.types import MaybeList

__all__ = [
    "T5LayerNorm",
    "MultiheadRPRAttention",
    "read_t5_gin_config_file"
]

IMPORTANT_PARAMS = ('d_ff',
                    'd_kv',
                    'd_model',
                    'dropout',
                    'num_heads',
                    'num_layers',
                    'inputs_length'
                    )


def read_t5_gin_config_file(config_file_path: str) -> Dict:
    r"""Simple helper function to read a gin file
    and get hyperparameters for T5.

    Args:
        config_file_path: path of config.gin file as a string.

    Returns:
        A dictionary with important parameters for loading T5.

    """
    config = {}

    with open(config_file_path, 'r') as gin_file:
        for line in gin_file:
            if line.startswith(IMPORTANT_PARAMS):
                assignment = line.strip().split()
                assert len(assignment) == 3
                arg_name, _, value = assignment
                config[arg_name] = ast.literal_eval(value)

    return config


class T5LayerNorm(nn.Module):
    r""" Custom LayerNorm for T5 with no mean subtraction and no bias.
    """

    def __init__(self, input_size: int, eps: float = 1e-5):
        super().__init__()

        self.w = nn.Parameter(torch.ones(input_size))
        self.eps = eps

    def forward(self,  # type: ignore
                x: torch.Tensor):
        x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * x


class MultiheadRPRAttention(ModuleBase):
    r"""
    A variation of MultiheadAttention introduced by

    Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. "Self-attention with
    relative position representations".  https://arxiv.org/pdf/1803.02155.pdf

    wherein it uses an alternative means of encoding positional information
    of an input sequence by learning an embedding which is stored in the
    first layer of Encoder/Decoder Stack but is shared amongst all attention
    layers in the stack.

    Args:
        input_size (int): Number of hidden dimension

        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

        stores_relative_position (bool): If the instance stores the learned
        realtive position embeddings.

    .. document private functions
    """

    def __init__(self, input_size: int, hparams=None,
                 stores_relative_position: bool = False):
        super().__init__(hparams=hparams)

        use_bias = self._hparams.use_bias
        self.is_decoder = self._hparams.is_decoder
        self.stores_rpr = stores_relative_position
        self.relative_attention_num_buckets = \
            self._hparams.relative_attention_num_buckets

        self.Q_dense = nn.Linear(input_size, self._hparams.num_units,
                                 bias=use_bias)
        self.K_dense = nn.Linear(input_size, self._hparams.num_units,
                                 bias=use_bias)
        self.V_dense = nn.Linear(input_size, self._hparams.num_units,
                                 bias=use_bias)
        self.O_dense = nn.Linear(self._hparams.num_units,
                                 self._hparams.output_dim, bias=use_bias)

        if self.stores_rpr:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self._hparams.num_heads)

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
                "name": "multihead_attention",
                "is_decoder": False,
                "relative_attention_num_buckets": 32
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

        `"is_decoder"`: bool
            To pass in if the attention is for a encoder or decoder block.

        `"name"`: relative_attention_num_buckets
            If the Attention mechnanism needs to use relative positional
            attention bias, then this hparam stores the relative attention
            num buckets.
        """
        return {
            'initializer': None,
            'num_heads': 8,
            'output_dim': 512,
            'num_units': 512,
            'dropout_rate': 0.1,
            'use_bias': False,
            'name': 'multihead_attention_rpr',
            'is_decoder': False,
            'relative_attention_num_buckets': 32
        }

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/master/
        mesh_tensorflow/transformer/transformer_layers.py#L595

        Translate relative position to a bucket number for relative attention.

        The relative position is defined as memory_position - query_position,
        i.e. the distance in tokens from the attending position to the
        attended-to position.  If bidirectional=False, then positive relative
        positions are invalid.

        We use smaller buckets for small absolute relative_position and
        larger buckets for larger absolute relative_positions.

        All relative positions >=max_distance map to the same bucket.
        All relative positions <=-max_distance map to the same bucket.

        This should allow for more graceful generalization to longer
        sequences than the model has been trained on.

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = (n < max_exact)

        # The other half of the buckets are for logarithmically bigger bins
        # in positions up to max_distance
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact) * (
                            num_buckets - max_exact)).to(torch.long)
        val_if_large = torch.min(val_if_large,
                                 torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]

        relative_position = memory_position - context_position
        #  [length_query, length_key]

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets)
        # [length_query, length_key]

        values = self.relative_attention_bias(rp_bucket)
        # [length_query, length_key, num_heads]

        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values  # [1, num_heads, length_query, length_key]

    def forward(self,  # type: ignore
                queries: torch.Tensor,
                memory: torch.Tensor,
                memory_attention_bias: torch.Tensor,
                cache: Optional[LayerCache] = None,
                position_bias: Optional[torch.Tensor] = None
                ) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Encodes the inputs.

        Args:
            queries: A 3D tensor with shape of
                ``[batch, length_query, depth_query]``.
            memory: A 4D tensor with shape of
                ``[batch, length_key, depth_key]``.
            memory_attention_bias: A 4D tensor with shape of
                ``[batch, length_key, num_units]``.
            cache: Memory cache only when inferring the sentence from scratch.
            position_bias: A 4D Tensor with shape of
                ``[batch, num_heads, length_query, length_query)]``.

        Returns:
            A tuple of 2 tensors, first, a tensor of shape
            ``[batch_size, max_time, dim]`` containing the encoded vectors
            and second a tensor of shape
            [batch, num_heads, length_query, length_query)] for sharing the
            position bias
        """
        length_query = queries.size(1)
        if memory is None:
            length_key = length_query  # Self-Attention
        else:
            length_key = memory.size(1)

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
                    res: MaybeList[torch.Tensor] = cache[  # type: ignore
                        key]  # type: ignore
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

        # All of the above [batch_size, num_heads, seq_length, memory_depth]
        # Q_ *= key_depth_per_head ** -0.5  # T5 does not scale

        logits = torch.einsum('bnqd,bnkd->bnqk', Q_, K_)  # type: ignore

        if position_bias is None:
            # Must compute bias using embedding stored.
            # Check if self.stores_rpr is True
            if not self.stores_rpr:
                raise ValueError("Layer must store embedding weights since"
                                 "relative bias not provided")
            position_bias = self.compute_bias(length_query, length_key)

            if memory_attention_bias is not None:
                memory_attention_bias = memory_attention_bias.to(
                    device=logits.device)
                position_bias = position_bias + memory_attention_bias

        logits += position_bias
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, self._hparams.dropout_rate,
                            self.training)
        outputs = torch.matmul(weights, V_)

        outputs = self._combine_heads(outputs)
        outputs = self.O_dense(outputs)
        # (batch_size, length_query, output_dim)

        return outputs, position_bias

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
