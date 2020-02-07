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

from typing import Optional, Union

import torch
from torch import nn

from texar.torch.modules.encoders.multihead_attention import Cache
from texar.torch.modules.pretrained.t5_utils import \
    T5LayerNorm, MultiheadRPRAttention
from texar.torch.modules.decoders.transformer_decoders import \
    TokenEmbedder, TokenPosEmbedder
from texar.torch.modules.encoders.transformer_encoder import \
    default_transformer_poswise_net_hparams
from texar.torch.modules.decoders.transformer_decoders \
    import TransformerDecoder
from texar.torch.modules.networks.networks import FeedForwardNetwork


class T5Decoder(TransformerDecoder):
    r"""T5 decoder that applies multi-head self-attention with relative
    position representation for sequence decoding.

    It is a stack of
    :class:`~texar.torch.modules.pretrained.t5_utilsMultiheadRPRAttention`,
    :class:`~texar.torch.modules.FeedForwardNetwork`, and residual connections.

    Args:
        token_embedder: An instance of :torch_nn:`Module`, or a function taking
            a :tensor:`LongTensor` ``tokens`` as argument. This is the embedder
            called in :meth:`embed_tokens` to convert input tokens to
            embeddings.
        token_pos_embedder: An instance of :torch_nn:`Module`, or a function
            taking two :tensor:`LongTensor`\ s ``tokens`` and ``positions`` as
            argument. This is the embedder called in :meth:`embed_tokens` to
            convert input tokens with positions to embeddings.

            .. note::
                Only one among :attr:`token_embedder` and
                :attr:`token_pos_embedder` should be specified. If neither is
                specified, you must subclass :class:`TransformerDecoder` and
                override :meth:`embed_tokens`.
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An output layer that transforms cell output
            to logits. This can be:

            - A callable layer, e.g., an instance of :torch_nn:`Module`.
            - A tensor. A :torch_nn:`Linear` layer will be created using the
              tensor as weights. The bias of the dense layer is determined
              by ``hparams.output_layer_bias``. This can be used to tie the
              output layer with the input embedding matrix, as proposed in
              https://arxiv.org/pdf/1608.05859.pdf.
            - `None`. A :torch_nn:`Linear` layer will be created based on
              :attr:`vocab_size` and ``hparams.output_layer_bias``.
            - If no output layer is needed at the end, set
              :attr:`vocab_size` to `None` and ``output_layer`` to
              :func:`~texar.torch.core.identity`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.

    .. document private functions
    """

    # State variables used during `dynamic_decode`. Assigned in `forward`.
    _state_max_decoding_length: int
    _state_context: Optional[torch.LongTensor]
    _state_context_sequence_length: Optional[torch.LongTensor]
    _state_cache: Cache

    def __init__(self,
                 token_embedder: Optional[TokenEmbedder] = None,
                 token_pos_embedder: Optional[TokenPosEmbedder] = None,
                 vocab_size: Optional[int] = None,
                 output_layer: Optional[Union[nn.Module, torch.Tensor]] = None,
                 hparams=None):
        super().__init__(
            token_embedder, token_pos_embedder,
            vocab_size=vocab_size, output_layer=output_layer, hparams=hparams)

        self.final_layer_norm = T5LayerNorm(self._input_size,  # type: ignore
                                            eps=self._hparams.eps)

    def initialize_blocks(self):
        r"""Helper function to initialize blocks.
        """
        for i in range(self._hparams.num_blocks):
            attn_module = MultiheadRPRAttention(
                self._input_size,
                self._hparams.multihead_attention,
                stores_relative_position=bool(i == 0))
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadRPRAttention should be equal "
                                 "to the dim of T5Decoder")
            self.self_attns.append(attn_module)
            self.self_attn_layer_norm.append(
                T5LayerNorm(self._input_size, eps=self._hparams.eps))

            attn_module = MultiheadRPRAttention(
                self._input_size, self._hparams.multihead_attention,
                stores_relative_position=bool(i == 0)
            )
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadRPRAttention should be equal "
                                 "to the dim of T5Decoder")
            self.enc_dec_attns.append(attn_module)
            self.end_dec_attn_layer_norm.append(
                T5LayerNorm(self._input_size, eps=self._hparams.eps))

            poswise_network = FeedForwardNetwork(
                hparams=self._hparams.poswise_feedforward)
            if (poswise_network.hparams.layers[-1]['kwargs']['out_features']
                    != self._hparams.dim):
                raise ValueError("The output dimension of "
                                 "FeedForwardNetwork should be equal "
                                 "to the dim of T5Decoder")
            self.poswise_networks.append(poswise_network)
            self.poswise_layer_norm.append(
                T5LayerNorm(self._input_size, eps=self._hparams.eps))

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # Same as in T5Encoder
                "num_blocks": 6,
                "dim": 512,
                "embedding_dropout": 0.1,
                "residual_dropout": 0.1,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                "multihead_attention": {
                    'name': 'multihead_rpr_attention',
                    'num_units': 512,
                    'output_dim': 512,
                    'num_heads': 8,
                    'dropout_rate': 0.1,
                    'use_bias': False,
                    'is_decoder': True,
                    'relative_attention_num_buckets': 32
                },
                "initializer": None,
                "eps": 1e-6,
                "name": "t5_decoder"

                # Additional for TransformerDecoder
                "embedding_tie": True,
                "output_layer_bias": False,
                "max_decoding_length": int(1e10),
            }

        Here:

        `"num_blocks"`: int
            Number of stacked blocks.

        `"dim"`: int
            Hidden dimension of the encoder.

        `"embedding_dropout"`: float
            Dropout rate of the input embedding.

        `"residual_dropout"`: float
            Dropout rate of the residual connections.

        `"poswise_feedforward"`: dict
            Hyperparameters for a feed-forward network used in residual
            connections.
            Make sure the dimension of the output tensor is equal to ``dim``.

            See
            :func:`~texar.torch.modules.default_transformer_poswise_net_hparams`
            for details.

        `"multihead_attention"`: dict
            Hyperparameters for the multi-head attention strategy.
            Make sure the ``output_dim`` in this module is equal to ``dim``.

            See :class:`~texar.torch.modules.MultiheadRPRAttention`
            for details.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.

            See :func:`~texar.torch.core.get_initializer` for details.

        `"embedding_tie"`: bool
            Whether to use the word embedding matrix as the output layer
            that computes logits. If `False`, a new dense layer is created.

        `"eps"`: float
            Epsilon values for layer norm layers.

        `"output_layer_bias"`: bool
            Whether to use bias to the output layer.

        `"max_decoding_length"`: int
            The maximum allowed number of decoding steps.
            Set to a very large number of avoid the length constraint.
            Ignored if provided in :meth:`forward` or ``"train_greedy"``
            decoding is used.

        `"name"`: str
            Name of the module.
        """
        dim = 512
        return {
            'num_blocks': 6,
            'dim': dim,
            'embedding_tie': True,
            'output_layer_bias': False,
            'max_decoding_length': int(1e10),
            'embedding_dropout': 0.1,
            'residual_dropout': 0.1,
            'poswise_feedforward': default_transformer_poswise_net_hparams(dim),
            'multihead_attention': {
                'name': 'multihead_rpr_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
                'is_decoder': True,
                'relative_attention_num_buckets': 32
            },
            'eps': 1e-6,
            'initializer': None,
            'name': "t5_decoder",
        }

    def _self_attention_stack(
            self, inputs: torch.Tensor,
            memory: Optional[torch.Tensor],
            decoder_self_attention_bias: Optional[torch.Tensor] = None,
            memory_attention_bias: Optional[torch.Tensor] = None,
            cache: Optional[Cache] = None
    ) -> torch.Tensor:
        r"""Forward through the stacked multi-head rpr attentions.
        """
        if cache is not None:
            if memory is not None:
                memory_attention_bias = cache['memory_attention_bias']
        else:
            assert decoder_self_attention_bias is not None

        x = self.embed_dropout(inputs)
        position_bias = None
        encdec_position_bias = None
        for i in range(self._hparams.num_blocks):
            layer_cache = cache['layers'][i] if cache is not None else None

            selfatt_output, position_bias = self.self_attns[i](
                queries=self.self_attn_layer_norm[i](x),
                memory=None,
                memory_attention_bias=decoder_self_attention_bias,
                cache=layer_cache,
                position_bias=position_bias
            )

            x = x + self.residual_dropout(selfatt_output)

            if memory is not None:
                encdec_output, encdec_position_bias = self.enc_dec_attns[i](
                    queries=self.end_dec_attn_layer_norm[i](x),
                    memory=memory,
                    memory_attention_bias=memory_attention_bias,
                    position_bias=encdec_position_bias
                )

                x = x + self.residual_dropout(encdec_output)

            sub_output = self.poswise_networks[i](self.poswise_layer_norm[i](x))
            x = x + self.residual_dropout(sub_output)

        return self.final_layer_norm(x)
