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
Utils of RoBERTa Modules.
"""

import os
from abc import ABC
from typing import Any, Dict

import torch

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedRoBERTaMixin",
]

_ROBERTA_PATH = "https://dl.fbaipublicfiles.com/fairseq/models/"


class PretrainedRoBERTaMixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the RoBERTa model.

    The RoBERTa model was proposed in (`Liu et al`. 2019)
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_.
    As a variant of the standard BERT model, RoBERTa trains for more
    iterations on more data with a larger batch size as well as other tweaks
    in pre-training. Differing from the standard BERT, the RoBERTa model
    does not use segmentation embedding. Available model names include:

      * ``roberta-base``: RoBERTa using the BERT-base architecture,
        125M parameters.
      * ``roberta-large``: RoBERTa using the BERT-large architecture,
        355M parameters.

    We provide the following RoBERTa classes:

      * :class:`~texar.torch.modules.RoBERTaEncoder` for text encoding.
      * :class:`~texar.torch.modules.RoBERTaClassifier` for text
        classification and sequence tagging.

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692
    """

    _MODEL_NAME = "RoBERTa"
    _MODEL2URL = {
        'roberta-base':
            _ROBERTA_PATH + "roberta.base.tar.gz",
        'roberta-large':
            _ROBERTA_PATH + "roberta.large.tar.gz",
    }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None

        for file in files:
            if file.endswith('model.pt'):
                config_path = os.path.join(root, file)
                args = torch.load(config_path, map_location="cpu")['args']
                hidden_dim = args.encoder_embed_dim
                vocab_size = 50265
                position_size = args.max_positions + 2
                embedding_dropout = args.dropout
                num_blocks = args.encoder_layers
                num_heads = args.encoder_attention_heads
                dropout_rate = args.attention_dropout
                residual_dropout = args.dropout
                intermediate_size = args.encoder_ffn_embed_dim
                hidden_act = args.activation_fn

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
            'decoder.sentence_encoder.embed_tokens.weight':
                'word_embedder._embedding',
            'decoder.sentence_encoder.embed_positions.weight':
                'position_embedder._embedding',
            'decoder.sentence_encoder.emb_layer_norm.weight':
                'encoder.input_normalizer.weight',
            'decoder.sentence_encoder.emb_layer_norm.bias':
                'encoder.input_normalizer.bias',
        }

        attention_tensor_map = {
            'final_layer_norm.weight':
                'encoder.output_layer_norm.{}.weight',
            'final_layer_norm.bias':
                'encoder.output_layer_norm.{}.bias',
            'fc1.weight':
                'encoder.poswise_networks.{}._layers.0.weight',
            'fc1.bias':
                'encoder.poswise_networks.{}._layers.0.bias',
            'fc2.weight':
                'encoder.poswise_networks.{}._layers.2.weight',
            'fc2.bias':
                'encoder.poswise_networks.{}._layers.2.bias',
            'self_attn_layer_norm.weight':
                'encoder.poswise_layer_norm.{}.weight',
            'self_attn_layer_norm.bias':
                'encoder.poswise_layer_norm.{}.bias',
            'self_attn.out_proj.weight':
                'encoder.self_attns.{}.O_dense.weight',
            'self_attn.out_proj.bias':
                'encoder.self_attns.{}.O_dense.bias',
            'self_attn.in_proj_weight': [
                'encoder.self_attns.{}.Q_dense.weight',
                'encoder.self_attns.{}.K_dense.weight',
                'encoder.self_attns.{}.V_dense.weight',
            ],
            'self_attn.in_proj_bias': [
                'encoder.self_attns.{}.Q_dense.bias',
                'encoder.self_attns.{}.K_dense.bias',
                'encoder.self_attns.{}.V_dense.bias'
            ],
        }

        checkpoint_path = os.path.abspath(os.path.join(cache_dir, 'model.pt'))
        device = next(self.parameters()).device
        params = torch.load(checkpoint_path, map_location=device)['model']

        for name, tensor in params.items():
            if name in global_tensor_map:
                v_name = global_tensor_map[name]
                pointer = self._name_to_variable(v_name)
                assert pointer.shape == tensor.shape
                pointer.data = tensor.data.type(pointer.dtype)
            elif name.startswith('decoder.sentence_encoder.layers.'):
                name = name.lstrip('decoder.sentence_encoder.layers.')
                layer_num, layer_name = name.split('.', 1)
                if layer_name in attention_tensor_map:
                    v_names = attention_tensor_map[layer_name]
                    if isinstance(v_names, str):
                        pointer = self._name_to_variable(
                            v_names.format(layer_num))
                        assert pointer.shape == tensor.shape
                        pointer.data = tensor.data.type(pointer.dtype)
                    else:
                        # Q, K, V in self-attention
                        tensors = torch.chunk(tensor, chunks=3, dim=0)
                        for i in range(3):
                            pointer = self._name_to_variable(
                                v_names[i].format(layer_num))
                            assert pointer.shape == tensors[i].shape
                            pointer.data = tensors[i].data.type(pointer.dtype)
                else:
                    raise NameError(f"Layer name '{layer_name}' not found")
