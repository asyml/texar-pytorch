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
GPT2 decoder.
"""

from typing import Optional

import torch

from texar.torch.modules.decoders.transformer_decoders import TransformerDecoder
from texar.torch.modules.embedders import PositionEmbedder, WordEmbedder
from texar.torch.modules.pretrained.pretrained_gpt2 import PretrainedGPT2Mixin

__all__ = [
    "GPT2Decoder",
]


class GPT2Decoder(TransformerDecoder, PretrainedGPT2Mixin):
    r"""Raw GPT2 Transformer for decoding sequences.

    This module basically stacks
    :class:`~texar.torch.modules.embedders.WordEmbedder`,
    :class:`~texar.torch.modules.embedders.PositionEmbedder`,
    :class:`~texar.torch.modules.encoders.TransformerDecoder`.

    This module supports the architecture first proposed
    in `(Radford et al.)` GPT2.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``117M``). Please refer to
            :class:`~texar.torch.modules.pretrained.PretrainedGPT2Mixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
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

        # The GPT2 decoder (a TransformerDecoder)
        super().__init__(vocab_size=self._hparams.vocab_size,
                         output_layer=word_embedder.embedding,
                         hparams=None)

        # Register modules after `__init__` is called.
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.init_pretrained_weights()

    def embed_tokens(self, tokens: torch.LongTensor,
                     positions: torch.LongTensor) -> torch.Tensor:
        word_embeds = self.word_embedder(tokens)
        pos_embeds = self.position_embedder(positions)
        return word_embeds + pos_embeds

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        * The decoder arch is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the encoder arch is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the encoder arch is defined by the
          configurations in `hparams` and weights are randomly initialized.

        .. code-block:: python

            {
                "name": "gpt2_decoder",
                "pretrained_model_name": "117M",
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

                # hparams for TransformerDecoder
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
                        "dropout_rate": 0.0,
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
            }

        Here:

        The default parameters are values for 117M GPT2 model.

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

        `"name"`: str
            Name of the module.
        """
        return {
            **TransformerDecoder.default_hparams(),
            'dim': 768,
            'num_blocks': 12,
            'use_gpt_config': True,
            'embedding_dropout': 0,
            'residual_dropout': 0,
            'multihead_attention': {
                'use_bias': True,
                'num_units': 768,
                'num_heads': 12,
                "dropout_rate": 0.0,
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

            'pretrained_model_name': '117M',
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

            'name': 'gpt2_decoder',
            '@no_typecheck': ['pretrained_model_name'],
        }
