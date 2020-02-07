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
T5 Model.
"""

from typing import Optional, Union

import torch

from texar.torch.core import layers, Identity
from texar.torch.modules.embedders.embedders import WordEmbedder
from texar.torch.modules.encoder_decoders.encoder_decoder_base\
    import EncoderDecoderBase
from texar.torch.modules.encoders import T5Encoder
from texar.torch.modules.decoders import T5Decoder
from texar.torch.modules.pretrained.t5 import PretrainedT5Mixin

__all__ = [
    "T5EncoderDecoder"
]


class T5EncoderDecoder(EncoderDecoderBase, PretrainedT5Mixin):
    r"""The pre-trained T5 model. Please see
    :class:`~texar.torch.modules.PretrainedT5Mixin` for a brief description
    of T5.

    This module basically stacks
    :class:`~texar.torch.modules.WordEmbedder`,
    :class:`~texar.torch.modules.T5Encoder`, and
    :class:`~texar.torch.modules.T5Decoder`.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``T5-Small``). Please refer to
            :class:`~texar.torch.modules.PretrainedT5Mixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        super().__init__(hparams=hparams)

        self.load_pretrained_config(pretrained_model_name, cache_dir)

        # Word embedding
        self.word_embedder = WordEmbedder(
            vocab_size=self._hparams.vocab_size,
            hparams=self._hparams.embed)

        # The encoder (a TransformerEncoder)
        self.encoder = T5Encoder(hparams=self._hparams.encoder)

        # The decoder (a TransformerDecoder)
        self.decoder = T5Decoder(
            token_embedder=self._embedding_fn,
            output_layer=Identity(),
            hparams=self._hparams.decoder)

        self.init_pretrained_weights()

    def _embedding_fn(self, tokens: torch.LongTensor
                      ) -> torch.Tensor:
        word_embed = self.word_embedder(tokens)
        return word_embed

    def reset_parameters(self):
        initialize = layers.get_initializer(self._hparams.initializer)
        if initialize is not None:
            # Do not re-initialize LayerNorm modules.
            for name, param in self.named_parameters():
                if name.split('.')[-1] == 'weight' and 'layer_norm' not in name:
                    initialize(param)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        * The model arch is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the model arch is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the encoder arch is defined by the
          configurations in `hparams` and weights are randomly initialized.

        .. code-block:: python

            {
                "pretrained_model_name": "T5-Small",
                "embed": {
                    "dim": 768,
                    "name": "word_embeddings"
                },
                "vocab_size": 32128,

                "encoder": {
                    "dim": 768,
                    "embedding_dropout": 0.1,
                    "multihead_attention": {
                        "dropout_rate": 0.1,
                        "name": "self",
                        "num_heads": 12,
                        "num_units": 768,
                        "output_dim": 768,
                        "use_bias": False,
                        "is_decoder": False,
                        "relative_attention_num_buckets": 32,
                    },
                    "eps": 1e-6,
                    "name": "encoder",
                    "num_blocks": 12,
                    "poswise_feedforward": {
                        "layers": [
                            {
                                "kwargs": {
                                    "in_features": 768,
                                    "out_features": 3072,
                                    "bias": False
                                },
                                "type": "Linear"
                            },
                            {"type": "ReLU"},
                            {
                                "kwargs": {
                                    "in_features": 3072,
                                    "out_features": 768,
                                    "bias": False
                                },
                                "type": "Linear"
                            }
                        ]
                    },
                    "residual_dropout": 0.1,
                    },

                "decoder": {
                    "eps": 1e-6,
                    "dim": 768,
                    "embedding_dropout": 0.1,
                    "multihead_attention": {
                        "dropout_rate": 0.1,
                        "name": "self",
                        "num_heads": 12,
                        "num_units": 768,
                        "output_dim": 768,
                        "use_bias": False,
                        "is_decoder": True,
                        "relative_attention_num_buckets": 32,
                    },
                    "name": "decoder",
                    "num_blocks": 12,
                    "poswise_feedforward": {
                        "layers": [
                            {
                                "kwargs": {
                                    "in_features": 768,
                                    "out_features": 3072,
                                    "bias": False
                                },
                                "type": "Linear"
                            },
                            {"type": "ReLU"},
                            {
                                "kwargs": {
                                    "in_features": 3072,
                                    "out_features": 768,
                                    "bias": False
                                },
                                "type": "Linear"
                            }
                        ]
                    },
                    "residual_dropout": 0.1,
                    },
                "hidden_size": 768,
                "initializer": None,
                "name": "t5_encoder_decoder",
            }

        Here:

        The default parameters are values for T5-Small model.

        `"pretrained_model_name"`: str or None
            The name of the pre-trained T5 model. If None, the model
            will be randomly initialized.

        `"embed"`: dict
            Hyperparameters for word embedding layer.

        `"vocab_size"`: int
            The vocabulary size of `inputs` in T5 model.

        `"encoder"`: dict
            Hyperparameters for the `T5Encoder`.
            See :func:`~texar.torch.modules.T5Encoder.default_hparams`
            for details.

        `"decoder"`: dict
            Hyperparameters for the `T5Decoder`.
            See :func:`~texar.torch.modules.T5Decoder.default_hparams`
            for details.

        `"hidden_size"`: int
            Size of the hidden layer.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.torch.core.get_initializer` for details.

        `"name"`: str
            Name of the module.
        """

        return {
            'pretrained_model_name': 'T5-Small',
            'embed': {
                'dim': 768,
                'name': 'word_embeddings'
            },
            'vocab_size': 32128,

            'encoder': {
                'dim': 768,
                'embedding_dropout': 0.1,
                'multihead_attention': {
                    'dropout_rate': 0.1,
                    'name': 'self',
                    'num_heads': 12,
                    'num_units': 768,
                    'output_dim': 768,
                    'use_bias': False,
                    'is_decoder': False,
                    'relative_attention_num_buckets': 32
                },
                'eps': 1e-6,
                'name': 'encoder',
                'num_blocks': 12,
                'poswise_feedforward': {
                    'layers': [
                        {
                            'kwargs': {
                                'in_features': 768,
                                'out_features': 3072,
                                'bias': False
                            },
                            'type': 'Linear'
                        },
                        {"type": "ReLU"},
                        {
                            'kwargs': {
                                'in_features': 3072,
                                'out_features': 768,
                                'bias': False
                            },
                            'type': 'Linear'
                        }
                    ]
                },
                'residual_dropout': 0.1,
            },
            'decoder': {
                'eps': 1e-6,
                'dim': 768,
                'embedding_dropout': 0.1,
                'multihead_attention': {
                    'dropout_rate': 0.1,
                    'name': 'self',
                    'num_heads': 12,
                    'num_units': 768,
                    'output_dim': 768,
                    'use_bias': False,
                    'is_decoder': True,
                    'relative_attention_num_buckets': 32
                },
                'name': 'decoder',
                'num_blocks': 12,
                'poswise_feedforward': {
                    'layers': [
                        {
                            'kwargs': {
                                'in_features': 768,
                                'out_features': 3072,
                                'bias': False
                            },
                            'type': 'Linear'
                        },
                        {"type": "ReLU"},
                        {
                            'kwargs': {
                                'in_features': 3072,
                                'out_features': 768,
                                'bias': False
                            },
                            'type': 'Linear'
                        }
                    ]
                },
                'residual_dropout': 0.1,
            },
            'hidden_size': 768,
            'initializer': None,
            'name': 't5_encoder_decoder',
            '@no_typecheck': ['pretrained_model_name']
        }

    def forward(self,  # type: ignore
                inputs: Union[torch.Tensor, torch.LongTensor],
                sequence_length: Optional[torch.LongTensor] = None):
        r"""Performs encoding and decoding.

        Args:
            inputs: Either a **2D Tensor** of shape ``[batch_size, max_time]``,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
            sequence_length: A 1D :tensor:`Tensor` of shape
                ``[batch_size]``. Input tokens beyond respective sequence
                lengths are masked out automatically.

        Returns:
            A pair :attr:`(encoder_output, decoder_output)`

            - :attr:`encoder_output`: A Tensor of shape
              `[batch_size, max_time, dim]` containing the encoded vectors.

            - :attr:`decoder_output`: An instance of
              :class:`~texar.torch.modules.TransformerDecoderOutput` which
              contains `sample_id` and `logits`.
        """
        if inputs.dim() == 2:
            word_embeds = self.word_embedder(ids=inputs)
        elif inputs.dim() == 3:
            word_embeds = self.word_embedder(soft_ids=inputs)
        else:
            raise ValueError("'inputs' should be a 2D or 3D tensor.")

        batch_size = inputs.size(0)

        if sequence_length is None:
            sequence_length = inputs.new_full(
                (batch_size,), inputs.size(1), dtype=torch.long)

        encoder_output = self.encoder(inputs=word_embeds,
                                      sequence_length=sequence_length)

        decoder_output = self.decoder(inputs=inputs,
                                      memory=encoder_output,
                                      memory_sequence_length=sequence_length)

        return encoder_output, decoder_output

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output of the encoder.
        """
        return self._hparams.hidden_size
