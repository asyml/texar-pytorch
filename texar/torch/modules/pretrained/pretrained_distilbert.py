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
Utils of DistilBERT Modules.
"""

import json
import os
from abc import ABC
from typing import Any, Dict

import torch

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedDistilBERTMixin",
]

_DISTILBERT_PATH = "https://s3.amazonaws.com/models.huggingface.co/bert/"


class PretrainedDistilBERTMixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implements the DistilBERT model.

    The DistilBERT model was proposed in (`Sanh et al.`. 2019)
    `Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT`_
    . A distilled model trained from the supervision of BERT. Available model
    names include:

      * ``distilbert-base-uncased``: 6-layer, 768-hidden. 12 heads,
        66M parameters.
      * ``distilbert-base-uncased-distilled-squad``: A fine-tuned version of
        ``distilbert-base-uncased`` fine tuned using knowledge distillation
        on SQuAD 1.0.

    We provide the following DistilBERT classes:

      * :class:`~texar.torch.modules.DistilBERTEncoder` for text encoding.
      * :class:`~texar.torch.modules.DistilBERTClassifier` for text
        classification and sequence tagging.

    .. _`Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT`:
        https://medium.com/huggingface/distilbert-8cf3380435b5
    """

    _MODEL_NAME = "DistilBERT"
    _MODEL2URL = {
        'distilbert-base-uncased': [
            _DISTILBERT_PATH + 'distilbert-base-uncased-pytorch_model.bin',
            _DISTILBERT_PATH + 'distilbert-base-uncased-config.json',
            _DISTILBERT_PATH + 'bert-base-uncased-vocab.txt',
        ],
        'distilbert-base-uncased-distilled-squad': [
            _DISTILBERT_PATH +
            'distilbert-base-uncased-distilled-squad-pytorch_model.bin',
            _DISTILBERT_PATH +
            'distilbert-base-uncased-distilled-squad-config.json',
            _DISTILBERT_PATH + 'bert-large-uncased-vocab.txt',
        ],
    }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None

        for file in files:
            if file.endswith('config.json'):
                config_path = os.path.join(root, file)
                with open(config_path) as f:
                    config_ckpt = json.loads(f.read())

        if config_path is None:
            raise ValueError(f"Cannot find the config file in {cache_dir}")

        configs = {
            'hidden_size': config_ckpt['dim'],
            'embed': {
                'name': 'word_embeddings',
                'dim': config_ckpt['dim']
            },
            'vocab_size': config_ckpt['vocab_size'],
            'use_sinusoidal_pos_embed': config_ckpt['sinusoidal_pos_embds'],
            'position_embed': {
                'name': 'position_embeddings',
                'dim': config_ckpt['dim']
            },
            'position_size': config_ckpt['max_position_embeddings'],
            'encoder': {
                'name': 'encoder',
                'embedding_dropout': config_ckpt['dropout'],
                'num_blocks': config_ckpt['n_layers'],
                'multihead_attention': {
                    'use_bias': True,
                    'num_units': config_ckpt['dim'],
                    'num_heads': config_ckpt['n_heads'],
                    'output_dim': config_ckpt['dim'],
                    'dropout_rate': config_ckpt['attention_dropout'],
                    'name': 'self'
                },
                'residual_dropout': config_ckpt['dropout'],
                'dim': config_ckpt['dim'],
                'use_bert_config': True,
                'poswise_feedforward': {
                    "layers": [{
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': config_ckpt['dim'],
                            'out_features': config_ckpt['hidden_dim'],
                            'bias': True,
                        }
                    }, {
                        'type': 'Bert' + config_ckpt['activation'].upper()
                    }, {
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': config_ckpt['hidden_dim'],
                            'out_features': config_ckpt['dim'],
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
            'distilbert.embeddings.word_embeddings.weight':
                'word_embedder._embedding',
            'distilbert.embeddings.position_embeddings.weight':
                'position_embedder.signal',
            'distilbert.embeddings.LayerNorm.weight':
                'encoder.input_normalizer.weight',
            'distilbert.embeddings.LayerNorm.bias':
                'encoder.input_normalizer.bias',
        }

        attention_tensor_map = {
            'attention.q_lin.weight': 'encoder.self_attns.{}.Q_dense.weight',
            'attention.q_lin.bias': 'encoder.self_attns.{}.Q_dense.bias',
            'attention.k_lin.weight': 'encoder.self_attns.{}.K_dense.weight',
            'attention.k_lin.bias': 'encoder.self_attns.{}.K_dense.bias',
            'attention.v_lin.weight': 'encoder.self_attns.{}.V_dense.weight',
            'attention.v_lin.bias': 'encoder.self_attns.{}.V_dense.bias',
            'attention.out_lin.weight': 'encoder.self_attns.{}.O_dense.weight',
            'attention.out_lin.bias': 'encoder.self_attns.{}.O_dense.bias',
            'sa_layer_norm.weight': 'encoder.poswise_layer_norm.{}.weight',
            'sa_layer_norm.bias': 'encoder.poswise_layer_norm.{}.bias',
            'ffn.lin1.weight': 'encoder.poswise_networks.{}._layers.0.weight',
            'ffn.lin1.bias': 'encoder.poswise_networks.{}._layers.0.bias',
            'ffn.lin2.weight': 'encoder.poswise_networks.{}._layers.2.weight',
            'ffn.lin2.bias': 'encoder.poswise_networks.{}._layers.2.bias',
            'output_layer_norm.weight': 'encoder.output_layer_norm.{}.weight',
            'output_layer_norm.bias': 'encoder.output_layer_norm.{}.bias',
        }

        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        checkpoint_path = None

        for file in files:
            if file.endswith('model.bin'):
                checkpoint_path = os.path.join(root, file)

        assert checkpoint_path is not None
        device = next(self.parameters()).device
        params = torch.load(checkpoint_path, map_location=device)

        for name, tensor in params.items():
            if name in global_tensor_map:
                v_name = global_tensor_map[name]
                pointer = self._name_to_variable(v_name)
                assert pointer.shape == tensor.shape
                pointer.data = tensor.data.type(pointer.dtype)
            elif name.startswith('distilbert.transformer.layer.'):
                name = name.lstrip('distilbert.transformer.layer.')
                layer_num, layer_name = name[0], name[2:]
                if layer_name in attention_tensor_map:
                    v_names = attention_tensor_map[layer_name]
                    pointer = self._name_to_variable(
                        v_names.format(layer_num))
                    assert pointer.shape == tensor.shape
                    pointer.data = tensor.data.type(pointer.dtype)
