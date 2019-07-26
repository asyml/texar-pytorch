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

from texar.core import layers
from texar.hyperparams import HParams
from texar.modules.pretrained.gpt2_utils import init_gpt2_checkpoint
from texar.modules.pretrained.pretrained_base import PretrainedBase
from texar.modules.embedders.embedders import WordEmbedder
from texar.modules.embedders.position_embedders import PositionEmbedder
from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.decoders.transformer_decoders import TransformerDecoder


__all__ = [
    "GPT2Encoder",
]


class GPT2Encoder(PretrainedBase, EncoderBase):
    r"""Raw GPT2 Transformer for encoding sequences.

    This module basically stacks
    :class:`~texar.modules.embedders.WordEmbedder`,
    :class:`~texar.modules.embedders.PositionEmbedder`,
    :class:`~texar.modules.encoders.TransformerDecoder`.

    This module supports the architecture first proposed
    in `(Radford et al.)` GPT2.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model to load selected in the list of:
            `117M`, `345M`. If `None`, will use the model name in
            :attr:`hparams`.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    model_name = "GPT2"

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):

        super().__init__(pretrained_model_name=pretrained_model_name,
                         cache_dir=cache_dir,
                         hparams=hparams)

        if self.pretrained_model_dir:
            self._hparams = HParams(self.pretrained_model_hparams,
                                    self._hparams.todict())

        # Word embedding
        self.word_embedder = WordEmbedder(
            vocab_size=self._hparams.vocab_size,
            hparams=self._hparams.embed)

        # Position embedding
        self.position_embedder = PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed)

        # The GPT2 encoder (a TransformerDecoder)
        self.decoder = TransformerDecoder(
            vocab_size=self._hparams.vocab_size,
            output_layer=layers.identity,
            hparams=self._hparams.decoder)

        if self.pretrained_model_dir:
            init_gpt2_checkpoint(self, self.pretrained_model_dir)
        elif self._hparams.initializer:
            initialize = layers.get_initializer(self._hparams.initializer)
            assert initialize is not None
            # Do not re-initialize LayerNorm modules.
            for name, param in self.named_parameters():
                if name.split('.')[-1] == 'weight' and 'layer_norm' not in name:
                    initialize(param)

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

        `"decoder"`: dict
            Hyperparameters for the TransformerDecoder.
            See :func:`~texar.modules.TransformerDecoder.default_harams`
            for details.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        `"name"`: str
            Name of the module.
        """
        return {
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

            'decoder': {
                'dim': 768,
                'num_blocks': 12,
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
                }
            },
            'initializer': None,
            'name': 'gpt2_encoder',
            '@no_typecheck': ['pretrained_model_name']
        }

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                sequence_length: Optional[torch.LongTensor] = None,
                **kwargs):
        r"""Encodes the inputs.

        Args:
            inputs: A 2D Tensor of shape `[batch_size, max_time]`,
                containing the token ids of tokens in the input sequences.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.
            **kwargs: Keyword arguments.

        Returns:
            outputs:  A Tensor of shape
            `[batch_size, max_time, dim]` containing the encoded vectors.
        """
        word_embeds = self.word_embedder(inputs)
        batch_size = inputs.shape[0]
        pos_length = inputs.new_full((batch_size,),
                                     inputs.shape[1],
                                     dtype=torch.int64)
        pos_embeds = self.position_embedder(sequence_length=pos_length)

        inputs_embeds = word_embeds + pos_embeds

        if sequence_length is None:
            sequence_length = inputs.new_full((batch_size,),
                                              inputs.shape[1],
                                              dtype=torch.int64)

        output = self.decoder(inputs=inputs_embeds,
                              sequence_length=sequence_length)

        return output.logits

    @property
    def output_size(self):
        return self._hparams.decoder.dim
