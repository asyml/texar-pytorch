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
GPT2 encoders.
"""

from typing import Optional

import torch

from texar.torch.modules.embedders.embedders import WordEmbedder
from texar.torch.modules.embedders.position_embedders import PositionEmbedder
from texar.torch.modules.encoders.transformer_encoder import TransformerEncoder
from texar.torch.modules.pretrained.gpt2 import PretrainedGPT2Mixin

__all__ = [
    "GPT2Encoder",
]


class GPT2Encoder(TransformerEncoder, PretrainedGPT2Mixin):
    r"""Raw GPT2 Transformer for encoding sequences.

    This module basically stacks
    :class:`~texar.torch.modules.WordEmbedder`,
    :class:`~texar.torch.modules.PositionEmbedder`,
    :class:`~texar.torch.modules.TransformerEncoder`.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``gpt2-small``). Please refer to
            :class:`~texar.torch.modules.PretrainedGPT2Mixin` for
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
        self.load_pretrained_config(pretrained_model_name, cache_dir, hparams)

        # Word embedding
        word_embedder = WordEmbedder(
            vocab_size=self._hparams.vocab_size,
            hparams=self._hparams.embed)

        # Position embedding
        position_embedder = PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed)

        # The GPT2 encoder (a TransformerEncoder)
        super().__init__(hparams=None)

        # Register modules after `__init__` is called.
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.init_pretrained_weights(load_output_layer=False)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        * The encoder arch is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the encoder arch is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the encoder arch is defined by the
          configurations in `hparams` and weights are randomly initialized.

        .. code-block:: python

            {
                "pretrained_model_name": "gpt2-small",
                "vocab_size": 50257,
                "context_size": 1024,
                "embedding_size": 768,
                "embed": {
                    "dim": 768,
                    "name": "word_embeddings"
                },
                "position_size": 1024,
                "position_embed": {
                    "dim": 768,
                    "name": "position_embeddings"
                },

                "decoder": {
                    "dim": 768,
                    "num_blocks": 12,
                    "use_gpt_config": True,
                    "embedding_dropout": 0,
                    "residual_dropout": 0,
                    "multihead_attention": {
                        "use_bias": True,
                        "num_units": 768,
                        "num_heads": 12,
                        "output_dim": 768
                    },
                    "initializer": {
                        "type": "variance_scaling_initializer",
                        "kwargs": {
                            "factor": 1.0,
                            "mode": "FAN_AVG",
                            "uniform": True
                        }
                    },
                    "poswise_feedforward": {
                        "layers": [
                            {
                                "type": "Linear",
                                "kwargs": {
                                    "in_features": 768,
                                    "out_features": 3072,
                                    "bias": True
                                }
                            },
                            {
                                "type": "GPTGELU",
                                "kwargs": {}
                            },
                            {
                                "type": "Linear",
                                "kwargs": {
                                    "in_features": 3072,
                                    "out_features": 768,
                                    "bias": True
                                }
                            }
                        ],
                        "name": "ffn"
                    }
                },
                "initializer": None,
                "name": "gpt2_encoder",
            }

        Here:

        The default parameters are values for 124M GPT2 model.

        `"pretrained_model_name"`: str or None
            The name of the pre-trained GPT2 model. If None, the model
            will be randomly initialized.

        `"embed"`: dict
            Hyperparameters for word embedding layer.

        `"vocab_size"`: int
            The vocabulary size of `inputs` in `GPT2Model`.

        `"position_embed"`: dict
            Hyperparameters for position embedding layer.

        `"position_size"`:  int
            The maximum sequence length that this model might ever be used with.

        `"decoder"`: dict
            Hyperparameters for the TransformerDecoder.
            See :func:`~texar.torch.modules.TransformerDecoder.default_hparams`
            for details.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.torch.core.get_initializer` for details.

        `"name"`: str
            Name of the module.
        """
        return {
            **TransformerEncoder.default_hparams(),
            'dim': 768,
            'num_blocks': 12,
            'use_bert_config': False,
            'use_gpt_config': True,
            'embedding_dropout': 0,
            'residual_dropout': 0,
            'multihead_attention': {
                'use_bias': True,
                'num_units': 768,
                'num_heads': 12,
                'output_dim': 768
            },
            'initializer': {
                'type': 'variance_scaling_initializer',
                'kwargs': {
                    'factor': 1.0,
                    'mode': 'FAN_AVG',
                    'uniform': True
                }
            },
            'poswise_feedforward': {
                'layers': [
                    {
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': 768,
                            'out_features': 3072,
                            'bias': True
                        }
                    },
                    {
                        'type': 'GPTGELU',
                        'kwargs': {}
                    },
                    {
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': 3072,
                            'out_features': 768,
                            'bias': True
                        }
                    }
                ],
                'name': 'ffn'
            },

            'pretrained_model_name': 'gpt2-small',
            'vocab_size': 50257,
            'context_size': 1024,
            'embedding_size': 768,
            'embed': {
                'dim': 768,
                'name': 'word_embeddings'
            },
            'position_size': 1024,
            'position_embed': {
                'dim': 768,
                'name': 'position_embeddings'
            },

            'name': 'gpt2_encoder',
            '@no_typecheck': ['pretrained_model_name'],
        }

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                sequence_length: Optional[torch.LongTensor] = None):
        r"""Encodes the inputs.

        Args:
            inputs: A 2D Tensor of shape `[batch_size, max_time]`,
                containing the token ids of tokens in the input sequences.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.

        Returns:
            outputs:  A Tensor of shape
            `[batch_size, max_time, dim]` containing the encoded vectors.
        """
        word_embeds = self.word_embedder(inputs)
        batch_size = inputs.size(0)
        pos_length = inputs.new_full(
            (batch_size,), inputs.size(1), dtype=torch.long)
        pos_embeds = self.position_embedder(sequence_length=pos_length)

        inputs_embeds = word_embeds + pos_embeds

        if sequence_length is None:
            sequence_length = inputs.new_full(
                (batch_size,), inputs.size(1), dtype=torch.long)

        output = super().forward(
            inputs=inputs_embeds, sequence_length=sequence_length)

        return output

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output.
        """
        return self._hparams.dim
