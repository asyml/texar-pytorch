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

import torch

from texar.torch.modules.encoders.transformer_encoder import TransformerEncoder
from texar.torch.modules.pretrained.t5_utils import \
    T5LayerNorm, MultiheadRPRAttention
from texar.torch.modules.networks.networks import FeedForwardNetwork
from texar.torch.modules.encoders.transformer_encoder import \
    default_transformer_poswise_net_hparams
from texar.torch.utils import sequence_mask, transformer_attentions as attn


class T5Encoder(TransformerEncoder):
    r"""Transformer based encoder that applies multi-head self attention with
    relative positional representations for encoding sequences for T5.

    This module basically stacks
    :class:`~texar.torch.modules.pretrained.t5_utils.MultiheadRPRAttention`,
    :class:`~texar.torch.modules.FeedForwardNetwork` and residual connections.
    This module supports the standard T5 architecture proposed in
    `(Raffel et al.) "Exploring the Limits of Transfer Learning with a Unified
    Text-to-Text Transformer"`.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    .. document private functions
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        self.final_layer_norm = T5LayerNorm(self._input_size,
                                            eps=self._hparams.eps)

    def initialize_blocks(self):
        r"""Helper function to initialize blocks.
        """
        for i in range(self._hparams.num_blocks):
            mh_attn = MultiheadRPRAttention(
                self._input_size,
                self._hparams.multihead_attention,
                stores_relative_position=bool(i == 0)
            )
            self.self_attns.append(mh_attn)

            self.self_attn_layer_norm.append(
                T5LayerNorm(self._input_size, eps=self._hparams.eps))
            if self._hparams.dim != mh_attn.hparams.output_dim:
                raise ValueError(
                    'The "dim" in the hparams of '
                    '"multihead_attention" should be equal to the '
                    '"dim" of T5Encoder')

            pw_net = FeedForwardNetwork(
                hparams=self._hparams['poswise_feedforward'])

            final_dim = pw_net.hparams.layers[-1]['kwargs']['out_features']
            if self._hparams.dim != final_dim:
                raise ValueError(
                    'The output dimenstion of '
                    '"poswise_feedforward" should be equal '
                    'to the "dim" of T5Encoder.')

            self.poswise_networks.append(pw_net)
            self.poswise_layer_norm.append(
                T5LayerNorm(self._input_size, eps=self._hparams.eps))

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "num_blocks": 6,
                "dim": 512,
                "embedding_dropout": 0.1,
                "residual_dropout": 0.1,
                "use_bert_config: False,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                'multihead_attention': {
                    'name': 'multihead_rpr_attention',
                    'num_units': 512,
                    'num_heads': 8,
                    'dropout_rate': 0.1,
                    'output_dim': 512,
                    'use_bias': False,
                    'is_decoder': False,
                    'relative_attention_num_buckets': 32
                },
                "initializer": None,
                "eps": 1e-6,
                "name": "t5_encoder"
            }

        Here:

        `"num_blocks"`: int
            Number of stacked blocks.

        `"dim"`: int
            Hidden dimension of the encoders.

        `"embedding_dropout"`: float
            Dropout rate of the input embedding.

        `"residual_dropout"`: float
            Dropout rate of the residual connections.

        "eps"`: float
            Epsilon values for layer norm layers.

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
            See :class:`~texar.torch.modules.MultiheadRPRAttention` for
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
            'embedding_dropout': 0.1,
            'residual_dropout': 0.1,
            'use_bert_config': False,
            'poswise_feedforward': default_transformer_poswise_net_hparams(dim),
            'multihead_attention': {
                'name': 'multihead_rpr_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
                'is_decoder': False,
                'relative_attention_num_buckets': 32
            },
            'initializer': None,
            'eps': 1e-6,
            'name': 't5_encoder',
        }

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                sequence_length: torch.LongTensor) -> torch.Tensor:
        r"""Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape ``[batch_size, max_time, dim]``,
                containing the embedding of input sequences. Note that
                the embedding dimension `dim` must equal `"dim"` in
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
        ignore_padding = attn.attention_bias_ignore_padding(inputs_padding)
        encoder_self_attention_bias = ignore_padding

        x = self.embed_dropout(inputs)

        position_bias = None

        for i in range(self._hparams.num_blocks):
            _queries_input = self.self_attn_layer_norm[i](x)

            attention_output, position_bias = self.self_attns[i](
                queries=_queries_input,
                memory=_queries_input,
                memory_attention_bias=encoder_self_attention_bias,
                position_bias=position_bias
            )

            attention_output = self.residual_dropout(attention_output)

            x = x + attention_output

            poswise_network = self.poswise_networks[i]
            poswise_normalizer = self.poswise_layer_norm[i]

            y = poswise_normalizer(x)

            original_shape = y.size()

            y = y.view(-1, self._hparams.dim)

            layer_output = poswise_network(y)
            sub_output = self.residual_dropout(layer_output)
            sub_output = sub_output.view(original_shape)

            x = x + sub_output

        x = self.final_layer_norm(x)

        return x
