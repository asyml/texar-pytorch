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

from typing import Any, Dict

import torch
from torch import nn

from texar.torch.core import layers
from texar.torch.modules.encoders.encoder_base import EncoderBase
from texar.torch.modules.encoders.multihead_attention import (
    MultiheadAttentionEncoder)
from texar.torch.modules.networks.networks import FeedForwardNetwork
from texar.torch.utils import transformer_attentions as attn
from texar.torch.utils.utils import sequence_mask

__all__ = [
    "default_transformer_poswise_net_hparams",
    "TransformerEncoder",
]


def default_transformer_poswise_net_hparams(input_dim: int,
                                            output_dim: int = 512) \
        -> Dict[str, Any]:
    r"""Returns default hyperparameters of a
    :class:`~texar.torch.modules.FeedForwardNetwork` as a position-wise network
    used in :class:`~texar.torch.modules.TransformerEncoder` and
    :class:`~texar.torch.modules.TransformerDecoder`.
    This is a 2-layer dense network with dropout in-between.

    .. code-block:: python

        {
            "layers": [
                {
                    "type": "Linear",
                    "kwargs": {
                        "in_features": input_dim,
                        "out_features": output_dim * 4,
                        "bias": True,
                    }
                },
                {
                    "type": "nn.ReLU",
                    "kwargs": {
                        "inplace": True
                    }
                },
                {
                    "type": "Dropout",
                    "kwargs": {
                        "p": 0.1,
                    }
                },
                {
                    "type": "Linear",
                    "kwargs": {
                        "in_features": output_dim * 4,
                        "out_features": output_dim,
                        "bias": True,
                    }
                }
            ],
            "name": "ffn"
        }

    Args:
        input_dim (int): The size of dense layer input.
        output_dim (int): The size of dense layer output.
    """
    return {
        "layers": [
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": input_dim,
                    "out_features": output_dim * 4,
                    "bias": True,
                }
            },
            {
                "type": "ReLU",
                "kwargs": {
                    "inplace": True
                }
            },
            {
                "type": "Dropout",
                "kwargs": {
                    "p": 0.1,
                }
            },
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": output_dim * 4,
                    "out_features": output_dim,
                    "bias": True,
                }
            }
        ],
        "name": "ffn"
    }


class TransformerEncoder(EncoderBase):
    r"""Transformer encoder that applies multi-head self attention for encoding
    sequences.

    This module basically stacks
    :class:`~texar.torch.modules.MultiheadAttentionEncoder`,
    :class:`~texar.torch.modules.FeedForwardNetwork` and residual connections.
    This module supports two types of architectures, namely, the standard
    Transformer Encoder architecture first proposed in
    `(Vaswani et al.) "Attention is All You Need"`, and
    the variant first used in `(Devlin et al.)` BERT. See
    :meth:`default_hparams` for the nuance between the two types of
    architectures.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    .. document private functions
    """

    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)
        self._input_size = self._hparams.dim
        self.self_attns = nn.ModuleList()
        if not self._hparams.use_bert_config:
            self.self_attn_layer_norm = nn.ModuleList()
        else:
            self.output_layer_norm = nn.ModuleList()
        self.poswise_networks = nn.ModuleList()
        self.poswise_layer_norm = nn.ModuleList()

        if self._hparams.use_bert_config:
            # In TensorFlow, eps for LayerNorm is 1e-12 by default.
            eps = 1e-12
        else:
            # In PyTorch, eps for LayerNorm is 1e-6 by default.
            eps = 1e-6

        for _ in range(self._hparams.num_blocks):
            mh_attn = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            self.self_attns.append(mh_attn)
            if not self._hparams.use_bert_config:
                self.self_attn_layer_norm.append(
                    nn.LayerNorm(self._input_size, eps=eps))
            if self._hparams.dim != mh_attn.hparams.output_dim:
                raise ValueError(
                    'The "dim" in the hparams of '
                    '"multihead_attention" should be equal to the '
                    '"dim" of TransformerEncoder')

            pw_net = FeedForwardNetwork(
                hparams=self._hparams['poswise_feedforward'])

            final_dim = pw_net.hparams.layers[-1]['kwargs']['out_features']
            if self._hparams.dim != final_dim:
                raise ValueError(
                    'The output dimenstion of '
                    '"poswise_feedforward" should be equal '
                    'to the "dim" of TransformerEncoder.')

            self.poswise_networks.append(pw_net)
            self.poswise_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=eps))
            if self._hparams.use_bert_config:
                self.output_layer_norm.append(
                    nn.LayerNorm(self._input_size, eps=eps))

        self.embed_dropout = nn.Dropout(p=self._hparams.embedding_dropout)
        self.residual_dropout = nn.Dropout(p=self._hparams.residual_dropout)

        if self._hparams.use_bert_config:
            self.input_normalizer = nn.LayerNorm(self._input_size, eps=eps)
        else:
            self.final_layer_norm = nn.LayerNorm(self._input_size, eps=eps)

        if self._hparams.initializer:
            initialize = layers.get_initializer(self._hparams.initializer)
            assert initialize is not None
            # Do not re-initialize LayerNorm modules.
            for name, param in self.named_parameters():
                if name.split('.')[-1] == 'weight' and 'layer_norm' not in name:
                    initialize(param)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "num_blocks": 6,
                "dim": 512,
                'use_bert_config': False,
                "embedding_dropout": 0.1,
                "residual_dropout": 0.1,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                'multihead_attention': {
                    'name': 'multihead_attention',
                    'num_units': 512,
                    'num_heads': 8,
                    'dropout_rate': 0.1,
                    'output_dim': 512,
                    'use_bias': False,
                },
                "initializer": None,
                "name": "transformer_encoder"
            }

        Here:

        `"num_blocks"`: int
            Number of stacked blocks.

        `"dim"`: int
            Hidden dimension of the encoders.

        `"use_bert_config"`: bool
            If `False`, apply the standard Transformer Encoder architecture from
            the original paper `(Vaswani et al.) "Attention is All You Need"`.
            If `True`, apply the Transformer Encoder architecture used in BERT
            `(Devlin et al.)` and the default setting of TensorFlow.
            The differences lie in:

            1. The standard arch restricts the word embedding of PAD token to
               all zero. The BERT arch does not.
            2. The attention bias for padding tokens: Standard architectures use
               ``-1e8`` for negative attention mask. BERT uses ``-1e4`` instead.
            3. The residual connections between internal tensors:
               In BERT, a residual layer connects the tensors *after* layer
               normalization. In standard architectures, the tensors are
               connected *before* layer normalization.

        `"embedding_dropout"`: float
            Dropout rate of the input embedding.

        `"residual_dropout"`: float
            Dropout rate of the residual connections.

        `"poswise_feedforward"`: dict
            Hyperparameters for a feed-forward network used in residual
            connections.
            Make sure the dimension of the output tensor is equal to ``"dim"``.
            See
            :func:`~texar.torch.modules.default_transformer_poswise_net_hparams`
            for details.

        `"multihead_attention"`: dict
            Hyperparameters for the multi-head attention strategy.
            Make sure the ``"output_dim"`` in this module is equal to ``"dim"``.
            See :class:`~texar.torch.modules.MultiheadAttentionEncoder` for
            details.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.torch.core.get_initializer` for details.

        `"name"`: str
            Name of the module.
        """
        dim = 512
        return {
            'num_blocks': 6,
            'dim': dim,
            'use_bert_config': False,
            'embedding_dropout': 0.1,
            'residual_dropout': 0.1,
            'poswise_feedforward': default_transformer_poswise_net_hparams(dim),
            'multihead_attention': {
                'name': 'multihead_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
            },
            'initializer': None,
            'name': 'transformer_encoder',
        }

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                sequence_length: torch.LongTensor) -> torch.Tensor:
        r"""Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape ``[batch_size, max_time, dim]``,
                containing the embedding of input sequences. Note that
                the embedding dimension `dim` must equal "dim" in
                :attr:`hparams`. The input embedding is typically an
                aggregation of word embedding and position embedding.
            sequence_length: A 1D :tensor:`LongTensor` of shape
                ``[batch_size]``. Input tokens beyond respective sequence
                lengths are masked out automatically.

        Returns:
            A Tensor of shape ``[batch_size, max_time, dim]`` containing the
            encoded vectors.
        """
        # Multiply input embedding with the sqrt of its dimension for
        # normalization

        inputs_padding = 1 - sequence_mask(
            sequence_length, inputs.size()[1]).float()
        if self._hparams.use_bert_config:
            ignore_padding = attn.attention_bias_ignore_padding(
                inputs_padding, bias_value=-1e4)
        else:
            ignore_padding = attn.attention_bias_ignore_padding(
                inputs_padding)
        encoder_self_attention_bias = ignore_padding

        input_embedding = inputs
        if self._hparams.use_bert_config:
            x = self.input_normalizer(input_embedding)
            x = self.embed_dropout(x)
        else:
            x = self.embed_dropout(input_embedding)

        for i in range(self._hparams.num_blocks):
            # trivial difference between BERT and original Transformer
            if self._hparams.use_bert_config:
                _queries_input = x
            else:
                _queries_input = self.self_attn_layer_norm[i](x)

            attention_output = self.self_attns[i](
                queries=_queries_input,
                memory=_queries_input,
                memory_attention_bias=encoder_self_attention_bias,
            )

            attention_output = self.residual_dropout(attention_output)

            x = x + attention_output

            poswise_network = self.poswise_networks[i]
            poswise_normalizer = self.poswise_layer_norm[i]

            if self._hparams.use_bert_config:
                x = poswise_normalizer(x)
                y = x
            else:
                y = poswise_normalizer(x)

            original_shape = y.size()

            y = y.view(-1, self._hparams.dim)

            layer_output = poswise_network(y)
            sub_output = self.residual_dropout(layer_output)
            sub_output = sub_output.view(original_shape)

            x = x + sub_output
            if self._hparams.use_bert_config:
                x = self.output_layer_norm[i](x)

        if not self._hparams.use_bert_config:
            x = self.final_layer_norm(x)
        return x

    @property
    def output_size(self) -> int:
        return self._hparams.dim
