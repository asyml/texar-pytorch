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

from typing import Dict, Optional, Tuple, Union

import torch

from texar.torch.modules.decoders.decoder_helpers import Helper
from texar.torch.modules.decoders.transformer_decoders import \
    TransformerDecoder, TransformerDecoderOutput
from texar.torch.modules.embedders import PositionEmbedder, WordEmbedder
from texar.torch.modules.pretrained.gpt2 import PretrainedGPT2Mixin

__all__ = [
    "GPT2Decoder",
]


class GPT2Decoder(PretrainedGPT2Mixin):
    r"""Raw GPT2 Transformer for decoding sequences. Please see
    :class:`~texar.torch.modules.PretrainedGPT2Mixin` for a brief description
    of GPT2.

    This module basically stacks
    :class:`~texar.torch.modules.WordEmbedder`,
    :class:`~texar.torch.modules.PositionEmbedder`,
    :class:`~texar.torch.modules.TransformerDecoder`.

    This module supports the architecture first proposed
    in `(Radford et al.)` GPT2.

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
    _IS_DECODE = True

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

        # Position embedding
        self.position_embedder = PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed)

        # The GPT2 decoder (a TransformerDecoder)
        def func(tokens, positions):
            word_embeds = self.word_embedder(tokens)
            pos_embeds = self.position_embedder(positions)
            return word_embeds + pos_embeds

        class GPT2TransformerDecoder(TransformerDecoder):
            def embed_tokens(self, tokens: torch.LongTensor,
                             positions: torch.LongTensor) -> torch.Tensor:
                return func(tokens, positions)

        self.decoder = GPT2TransformerDecoder(
            vocab_size=self._hparams.vocab_size,
            output_layer=self.word_embedder.embedding,
            hparams=self._hparams.decoder)

        self.init_pretrained_weights()

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

        `"name"`: str
            Name of the module.
        """
        return {
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

            'name': 'gpt2_decoder',
            '@no_typecheck': ['pretrained_model_name'],
        }

    def forward(self,  # type: ignore
                inputs: Optional[torch.Tensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                memory: Optional[torch.Tensor] = None,
                memory_sequence_length: Optional[torch.LongTensor] = None,
                memory_attention_bias: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_sequence_length: Optional[torch.LongTensor] = None,
                helper: Optional[Helper] = None,
                decoding_strategy: str = 'train_greedy',
                max_decoding_length: Optional[int] = None,
                impute_finished: bool = False,
                infer_mode: Optional[bool] = None,
                beam_width: Optional[int] = None,
                length_penalty: float = 0.,
                **kwargs) \
            -> Union[
                TransformerDecoderOutput,
                Tuple[TransformerDecoderOutput, torch.LongTensor],
                Dict[str, torch.Tensor]]:
        r"""Performs decoding. Has exact the same interfaces with
        :meth:`texar.torch.modules.TransformerDecoder.forward`. Please refer to
        it for the detailed usage.
        """
        return self.decoder(inputs=inputs,
                            sequence_length=sequence_length,
                            memory=memory,
                            memory_sequence_length=memory_sequence_length,
                            memory_attention_bias=memory_attention_bias,
                            context=context,
                            context_sequence_length=context_sequence_length,
                            helper=helper,
                            decoding_strategy=decoding_strategy,
                            max_decoding_length=max_decoding_length,
                            impute_finished=impute_finished,
                            infer_mode=infer_mode,
                            beam_width=beam_width,
                            length_penalty=length_penalty,
                            **kwargs)
