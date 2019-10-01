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
BERT encoder.
"""

from typing import Optional, Union

import torch
from torch import nn

from texar.torch.core import layers
from texar.torch.modules.embedders.embedders import WordEmbedder
from texar.torch.modules.embedders.position_embedders import PositionEmbedder
from texar.torch.modules.encoders.encoder_base import EncoderBase
from texar.torch.modules.encoders.transformer_encoder import TransformerEncoder
from texar.torch.modules.pretrained.bert import PretrainedBERTMixin

__all__ = [
    "BERTEncoder",
]


class BERTEncoder(EncoderBase, PretrainedBERTMixin):
    r"""Raw BERT Transformer for encoding sequences. Please see
    :class:`~texar.torch.modules.PretrainedBERTMixin` for a brief description
    of BERT.

    This module basically stacks
    :class:`~texar.torch.modules.WordEmbedder`,
    :class:`~texar.torch.modules.PositionEmbedder`,
    :class:`~texar.torch.modules.TransformerEncoder` and a dense
    pooler.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``bert-base-uncased``). Please refer to
            :class:`~texar.torch.modules.PretrainedBERTMixin` for
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

        # Segment embedding for each type of tokens
        self.segment_embedder = None
        if self._hparams.get('type_vocab_size', 0) > 0:
            self.segment_embedder = WordEmbedder(
                vocab_size=self._hparams.type_vocab_size,
                hparams=self._hparams.segment_embed)

        # Position embedding
        self.position_embedder = PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed)

        # The BERT encoder (a TransformerEncoder)
        self.encoder = TransformerEncoder(hparams=self._hparams.encoder)

        self.pooler = nn.Sequential(
            nn.Linear(self._hparams.hidden_size, self._hparams.hidden_size),
            nn.Tanh())

        self.init_pretrained_weights()

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
                "pretrained_model_name": "bert-base-uncased",
                "embed": {
                    "dim": 768,
                    "name": "word_embeddings"
                },
                "vocab_size": 30522,
                "segment_embed": {
                    "dim": 768,
                    "name": "token_type_embeddings"
                },
                "type_vocab_size": 2,
                "position_embed": {
                    "dim": 768,
                    "name": "position_embeddings"
                },
                "position_size": 512,

                "encoder": {
                    "dim": 768,
                    "embedding_dropout": 0.1,
                    "multihead_attention": {
                        "dropout_rate": 0.1,
                        "name": "self",
                        "num_heads": 12,
                        "num_units": 768,
                        "output_dim": 768,
                        "use_bias": True
                    },
                    "name": "encoder",
                    "num_blocks": 12,
                    "poswise_feedforward": {
                        "layers": [
                            {
                                "kwargs": {
                                    "in_features": 768,
                                    "out_features": 3072,
                                    "bias": True
                                },
                                "type": "Linear"
                            },
                            {"type": "BertGELU"},
                            {
                                "kwargs": {
                                    "in_features": 3072,
                                    "out_features": 768,
                                    "bias": True
                                },
                                "type": "Linear"
                            }
                        ]
                    },
                    "residual_dropout": 0.1,
                    "use_bert_config": True
                    },
                "hidden_size": 768,
                "initializer": None,
                "name": "bert_encoder",
            }

        Here:

        The default parameters are values for uncased BERT-Base model.

        `"pretrained_model_name"`: str or None
            The name of the pre-trained BERT model. If None, the model
            will be randomly initialized.

        `"embed"`: dict
            Hyperparameters for word embedding layer.

        `"vocab_size"`: int
            The vocabulary size of `inputs` in BERT model.

        `"segment_embed"`: dict
            Hyperparameters for segment embedding layer.

        `"type_vocab_size"`: int
            The vocabulary size of the `segment_ids` passed into `BertModel`.

        `"position_embed"`: dict
            Hyperparameters for position embedding layer.

        `"position_size"`: int
            The maximum sequence length that this model might ever be used with.

        `"encoder"`: dict
            Hyperparameters for the TransformerEncoder.
            See :func:`~texar.torch.modules.TransformerEncoder.default_hparams`
            for details.

        `"hidden_size"`: int
            Size of the pooler dense layer.

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.torch.core.get_initializer` for details.

        `"name"`: str
            Name of the module.
        """

        return {
            'pretrained_model_name': 'bert-base-uncased',
            'embed': {
                'dim': 768,
                'name': 'word_embeddings'
            },
            'vocab_size': 30522,
            'segment_embed': {
                'dim': 768,
                'name': 'token_type_embeddings'
            },
            'type_vocab_size': 2,
            'position_embed': {
                'dim': 768,
                'name': 'position_embeddings'
            },
            'position_size': 512,

            'encoder': {
                'dim': 768,
                'embedding_dropout': 0.1,
                'multihead_attention': {
                    'dropout_rate': 0.1,
                    'name': 'self',
                    'num_heads': 12,
                    'num_units': 768,
                    'output_dim': 768,
                    'use_bias': True
                },
                'name': 'encoder',
                'num_blocks': 12,
                'poswise_feedforward': {
                    'layers': [
                        {
                            'kwargs': {
                                'in_features': 768,
                                'out_features': 3072,
                                'bias': True
                            },
                            'type': 'Linear'
                        },
                        {"type": "BertGELU"},
                        {
                            'kwargs': {
                                'in_features': 3072,
                                'out_features': 768,
                                'bias': True
                            },
                            'type': 'Linear'
                        }
                    ]
                },
                'residual_dropout': 0.1,
                'use_bert_config': True
            },
            'hidden_size': 768,
            'initializer': None,
            'name': 'bert_encoder',
            '@no_typecheck': ['pretrained_model_name']
        }

    def forward(self,  # type: ignore
                inputs: Union[torch.Tensor, torch.LongTensor],
                sequence_length: Optional[torch.LongTensor] = None,
                segment_ids: Optional[torch.LongTensor] = None):
        r"""Encodes the inputs.

        Args:
            inputs: Either a **2D Tensor** of shape `[batch_size, max_time]`,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time]`, containing the segment ids
                of tokens in input sequences. If `None` (default), a
                tensor with all elements set to zero is used.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.

        Returns:
            A pair :attr:`(outputs, pooled_output)`

            - :attr:`outputs`:  A Tensor of shape
              `[batch_size, max_time, dim]` containing the encoded vectors.

            - :attr:`pooled_output`: A Tensor of size
              `[batch_size, hidden_size]` which is the output of a pooler
              pre-trained on top of the hidden state associated to the first
              character of the input (`CLS`), see BERT's paper.
        """
        if inputs.dim() == 2:
            word_embeds = self.word_embedder(ids=inputs)
        elif inputs.dim() == 3:
            word_embeds = self.word_embedder(soft_ids=inputs)
        else:
            raise ValueError("'inputs' should be a 2D or 3D tensor.")

        batch_size = inputs.size(0)
        pos_length = inputs.new_full((batch_size,), inputs.size(1),
                                     dtype=torch.int64)
        pos_embeds = self.position_embedder(sequence_length=pos_length)

        if self.segment_embedder is not None:
            if segment_ids is None:
                segment_ids = torch.zeros((inputs.size(0), inputs.size(1)),
                                          dtype=torch.long,
                                          device=inputs.device)
            segment_embeds = self.segment_embedder(segment_ids)
            inputs_embeds = word_embeds + segment_embeds + pos_embeds
        else:
            inputs_embeds = word_embeds + pos_embeds

        if sequence_length is None:
            sequence_length = inputs.new_full((batch_size,), inputs.size(1),
                                              dtype=torch.int64)

        output = self.encoder(inputs_embeds, sequence_length)

        # taking the hidden state corresponding to the first token.
        first_token_tensor = output[:, 0, :]

        pooled_output = self.pooler(first_token_tensor)

        return output, pooled_output

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output
        :attr:`pooled_output`.
        """
        return self._hparams.hidden_size
