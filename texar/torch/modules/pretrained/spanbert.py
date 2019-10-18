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
Utils of SpanBERT Modules.
"""

from typing import Any, Dict

import json
import os

from abc import ABC

import torch

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedSpanBERTMixin",
]

_SPANBERT_PATH = "https://dl.fbaipublicfiles.com/fairseq/models/"


class PretrainedSpanBERTMixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the SpanBERT model.

    The SpanBERT model was proposed in (`Joshi et al`. 2019)
    `SpanBERT: Improving Pre-training by Representing and Predicting Spans`_.
    As a variant of the standard BERT model, SpanBERT extends BERT by
    (1) masking contiguous random spans, rather than random tokens, and
    (2) training the span boundary representations to predict the entire
    content of the masked span, without relying on the individual token
    representations within it. Differing from the standard BERT, the
    SpanBERT model does not use segmentation embedding. Available model names
    include:

      * ``spanbert-base-cased``: SpanBERT using the BERT-base architecture,
        12-layer, 768-hidden, 12-heads , 110M parameters.
      * ``spanbert-large-cased``: SpanBERT using the BERT-large architecture,
        24-layer, 1024-hidden, 16-heads, 340M parameters.

    We provide the following SpanBERT classes:

      * :class:`~texar.torch.modules.SpanBERTEncoder` for text encoding.

    .. _`SpanBERT: Improving Pre-training by Representing and Predicting Spans`:
        https://arxiv.org/abs/1907.10529
    """

    _MODEL_NAME = "SpanBERT"
    _MODEL2URL = {
        'spanbert-base-cased':
            _SPANBERT_PATH + "spanbert_hf_base.tar.gz",
        'spanbert-large-cased':
            _SPANBERT_PATH + "spanbert_hf.tar.gz",
    }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None

        for file in files:
            if file == 'config.json':
                config_path = os.path.join(root, file)
                with open(config_path) as f:
                    config_ckpt = json.loads(f.read())
                    hidden_dim = config_ckpt['hidden_size']
                    vocab_size = config_ckpt['vocab_size']
                    position_size = config_ckpt['max_position_embeddings']
                    embedding_dropout = config_ckpt['hidden_dropout_prob']
                    num_blocks = config_ckpt['num_hidden_layers']
                    num_heads = config_ckpt['num_attention_heads']
                    dropout_rate = config_ckpt['attention_probs_dropout_prob']
                    residual_dropout = config_ckpt['hidden_dropout_prob']
                    intermediate_size = config_ckpt['intermediate_size']
                    hidden_act = config_ckpt['hidden_act']

        if config_path is None:
            raise ValueError(f"Cannot find the config file in {cache_dir}")

        configs = {
            'hidden_size': hidden_dim,
            'embed': {
                'name': 'word_embeddings',
                'dim': hidden_dim
            },
            'vocab_size': vocab_size,
            'position_embed': {
                'name': 'position_embeddings',
                'dim': hidden_dim
            },
            'position_size': position_size,
            'encoder': {
                'name': 'encoder',
                'embedding_dropout': embedding_dropout,
                'num_blocks': num_blocks,
                'multihead_attention': {
                    'use_bias': True,
                    'num_units': hidden_dim,
                    'num_heads': num_heads,
                    'output_dim': hidden_dim,
                    'dropout_rate': dropout_rate,
                    'name': 'self'
                },
                'residual_dropout': residual_dropout,
                'dim': hidden_dim,
                'use_bert_config': True,
                'poswise_feedforward': {
                    "layers": [{
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': hidden_dim,
                            'out_features': intermediate_size,
                            'bias': True,
                        }
                    }, {
                        'type': 'Bert' + hidden_act.upper()
                    }, {
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': intermediate_size,
                            'out_features': hidden_dim,
                            'bias': True,
                        }
                    }],
                },
            }
        }

        return configs

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str, **kwargs):
        global_tensor_map = {
            'bert.embeddings.word_embeddings.weight':
                'word_embedder._embedding',
            'bert.embeddings.position_embeddings.weight':
                'position_embedder._embedding',
            'bert.embeddings.LayerNorm.weight':
                'encoder.input_normalizer.weight',
            'bert.embeddings.LayerNorm.bias':
                'encoder.input_normalizer.bias',
        }

        attention_tensor_map = {
            "attention.self.key.bias": "self_attns.{}.K_dense.bias",
            "attention.self.query.bias": "self_attns.{}.Q_dense.bias",
            "attention.self.value.bias": "self_attns.{}.V_dense.bias",
            "attention.output.dense.bias": "self_attns.{}.O_dense.bias",
            "attention.output.LayerNorm.weight": "poswise_layer_norm.{}.weight",
            "attention.output.LayerNorm.bias": "poswise_layer_norm.{}.bias",
            "intermediate.dense.bias": "poswise_networks.{}._layers.0.bias",
            "output.dense.bias": "poswise_networks.{}._layers.2.bias",
            "output.LayerNorm.weight": "output_layer_norm.{}.weight",
            "output.LayerNorm.bias": "output_layer_norm.{}.bias",
            "attention.self.key.weight": "self_attns.{}.K_dense.weight",
            "attention.self.query.weight": "self_attns.{}.Q_dense.weight",
            "attention.self.value.weight": "self_attns.{}.V_dense.weight",
            "attention.output.dense.weight": "self_attns.{}.O_dense.weight",
            "intermediate.dense.weight": "poswise_networks.{}._layers.0.weight",
            "output.dense.weight": "poswise_networks.{}._layers.2.weight",
        }

        checkpoint_path = os.path.abspath(os.path.join(cache_dir,
                                                       'pytorch_model.bin'))
        device = next(self.parameters()).device
        params = torch.load(checkpoint_path, map_location=device)

        for name, tensor in params.items():
            if name in global_tensor_map:
                v_name = global_tensor_map[name]
                pointer = self._name_to_variable(v_name)
                assert pointer.shape == tensor.shape
                pointer.data = tensor.data.type(pointer.dtype)
            elif name.startswith('bert.encoder.layer.'):
                name = name.lstrip('bert.encoder.layer.')
                layer_num, layer_name = name.split('.', 1)
                if layer_name in attention_tensor_map:
                    v_name = attention_tensor_map[layer_name]
                    pointer = self._name_to_variable(
                        'encoder.' + v_name.format(layer_num))
                    assert pointer.shape == tensor.shape
                    pointer.data = tensor.data.type(pointer.dtype)
