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
RoBERTa encoder.
"""

from typing import Optional, Union

import torch

from texar.torch.modules.encoders.bert_encoder import BERTEncoder
from texar.torch.modules.pretrained.roberta import \
    PretrainedRoBERTaMixin

__all__ = [
    "RoBERTaEncoder",
]


class RoBERTaEncoder(PretrainedRoBERTaMixin, BERTEncoder):
    r"""RoBERTa Transformer for encoding sequences. Please see
    :class:`~texar.torch.modules.PretrainedRoBERTaMixin` for a brief description
    of RoBERTa.

    This module basically stacks
    :class:`~texar.torch.modules.WordEmbedder`,
    :class:`~texar.torch.modules.PositionEmbedder`,
    :class:`~texar.torch.modules.TransformerEncoder` and a dense
    pooler.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``roberta-base``). Please refer to
            :class:`~texar.torch.modules.PretrainedRoBERTaMixin` for
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
                "pretrained_model_name": "roberta-base",
                "embed": {
                    "dim": 768,
                    "name": "word_embeddings"
                },
                "vocab_size": 50265,
                "position_embed": {
                    "dim": 768,
                    "name": "position_embeddings"
                },
                "position_size": 514,

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
                "name": "roberta_encoder",
            }

        Here:

        The default parameters are values for RoBERTa-Base model.

        `"pretrained_model_name"`: str or None
            The name of the pre-trained RoBERTa model. If None, the model
            will be randomly initialized.

        `"embed"`: dict
            Hyperparameters for word embedding layer.

        `"vocab_size"`: int
            The vocabulary size of `inputs` in RoBERTa model.

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
            'pretrained_model_name': 'roberta-base',
            'embed': {
                'dim': 768,
                'name': 'word_embeddings'
            },
            'vocab_size': 50265,
            'position_embed': {
                'dim': 768,
                'name': 'position_embeddings'
            },
            'position_size': 514,

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
            'name': 'roberta_encoder',
            '@no_typecheck': ['pretrained_model_name']
        }

    def forward(self,  # type: ignore
                inputs: Union[torch.Tensor, torch.LongTensor],
                sequence_length: Optional[torch.LongTensor] = None,
                segment_ids: Optional[torch.LongTensor] = None):
        r"""Encodes the inputs. Differing from the standard BERT, the RoBERTa
        model does not use segmentation embedding. As a result, RoBERTa does not
        require `segment_ids` as an input.

        Args:
            inputs: Either a **2D Tensor** of shape `[batch_size, max_time]`,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
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
              character of the input (`CLS`), see RoBERTa's paper.
        """
        if segment_ids is not None:
            raise ValueError("segment_ids should be None in RoBERTaEncoder.")

        output, pooled_output = super().forward(inputs=inputs,
                                                sequence_length=sequence_length,
                                                segment_ids=None)

        return output, pooled_output
