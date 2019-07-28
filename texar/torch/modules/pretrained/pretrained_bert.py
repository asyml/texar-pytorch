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
Utils of BERT Modules.
"""

import json
import os
from abc import ABC
from typing import Any, Dict

import torch

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedBERTMixin",
]

_BERT_PATH = "https://storage.googleapis.com/bert_models/"


class PretrainedBERTMixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the BERT model.

    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by `Devlin et al.` It is a bidirectional Transformer model pre-trained
    on a large corpus.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805
    """

    _MODEL_NAME = "BERT"
    _MODEL2URL = {
        'bert-base-uncased':
            _BERT_PATH + "2018_10_18/uncased_L-12_H-768_A-12.zip",
        'bert-large-uncased':
            _BERT_PATH + "2018_10_18/uncased_L-24_H-1024_A-16.zip",
        'bert-base-cased':
            _BERT_PATH + "2018_10_18/cased_L-12_H-768_A-12.zip",
        'bert-large-cased':
            _BERT_PATH + "2018_10_18/cased_L-24_H-1024_A-16.zip",
        'bert-base-multilingual-uncased':
            _BERT_PATH + "2018_11_23/multi_cased_L-12_H-768_A-12.zip",
        'bert-base-multilingual-cased':
            _BERT_PATH + "2018_11_03/multilingual_L-12_H-768_A-12.zip",
        'bert-base-chinese':
            _BERT_PATH + "2018_11_03/chinese_L-12_H-768_A-12.zip",
    }

    @classmethod
    def _transform_config(cls, cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None
        for file in files:
            if file.endswith('config.json'):
                config_path = os.path.join(root, file)
        if config_path is None:
            raise ValueError(f"Cannot find the config file in {cache_dir}")

        with open(config_path) as f:
            config_ckpt = json.loads(f.read())

        hidden_dim = config_ckpt['hidden_size']
        configs = {
            'hidden_size': hidden_dim,
            'embed': {
                'name': 'word_embeddings',
                'dim': hidden_dim
            },
            'vocab_size': config_ckpt['vocab_size'],
            'segment_embed': {
                'name': 'token_type_embeddings',
                'dim': hidden_dim
            },
            'type_vocab_size': config_ckpt['type_vocab_size'],
            'position_embed': {
                'name': 'position_embeddings',
                'dim': hidden_dim
            },
            'position_size': config_ckpt['max_position_embeddings'],
            'encoder': {
                'name': 'encoder',
                'embedding_dropout': config_ckpt['hidden_dropout_prob'],
                'num_blocks': config_ckpt['num_hidden_layers'],
                'multihead_attention': {
                    'use_bias': True,
                    'num_units': hidden_dim,
                    'num_heads': config_ckpt['num_attention_heads'],
                    'output_dim': hidden_dim,
                    'dropout_rate': config_ckpt['attention_probs_dropout_prob'],
                    'name': 'self'
                },
                'residual_dropout': config_ckpt['hidden_dropout_prob'],
                'dim': hidden_dim,
                'use_bert_config': True,
                'poswise_feedforward': {
                    "layers": [{
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': hidden_dim,
                            'out_features': config_ckpt['intermediate_size'],
                            'bias': True,
                        }
                    }, {
                        'type': 'Bert' + config_ckpt['hidden_act'].upper()
                    }, {
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': config_ckpt['intermediate_size'],
                            'out_features': hidden_dim,
                            'bias': True,
                        }
                    }],
                },
            }
        }

        return configs

    def _init_from_checkpoint(self, cache_dir: str, **kwargs):
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            print("Loading TensorFlow models in PyTorch requires installing "
                  "TensorFlow. Please see https://www.tensorflow.org/install/ "
                  "for installation instructions.")
            raise

        global_tensor_map = {
            'bert/embeddings/word_embeddings': 'word_embedder._embedding',
            'bert/embeddings/token_type_embeddings':
                'segment_embedder._embedding',
            'bert/embeddings/position_embeddings':
                'position_embedder._embedding',
            'bert/embeddings/LayerNorm/beta':
                'encoder.input_normalizer.bias',
            'bert/embeddings/LayerNorm/gamma':
                'encoder.input_normalizer.weight',
        }
        layer_tensor_map = {
            "attention/self/key/bias": "self_attns.{}.K_dense.bias",
            "attention/self/query/bias": "self_attns.{}.Q_dense.bias",
            "attention/self/value/bias": "self_attns.{}.V_dense.bias",
            "attention/output/dense/bias": "self_attns.{}.O_dense.bias",
            "attention/output/LayerNorm/gamma": "poswise_layer_norm.{}.weight",
            "attention/output/LayerNorm/beta": "poswise_layer_norm.{}.bias",
            "intermediate/dense/bias": "poswise_networks.{}._layers.0.bias",
            "output/dense/bias": "poswise_networks.{}._layers.2.bias",
            "output/LayerNorm/gamma": "output_layer_norm.{}.weight",
            "output/LayerNorm/beta": "output_layer_norm.{}.bias",
        }
        layer_transpose_map = {
            "attention/self/key/kernel": "self_attns.{}.K_dense.weight",
            "attention/self/query/kernel": "self_attns.{}.Q_dense.weight",
            "attention/self/value/kernel": "self_attns.{}.V_dense.weight",
            "attention/output/dense/kernel": "self_attns.{}.O_dense.weight",
            "intermediate/dense/kernel": "poswise_networks.{}._layers.0.weight",
            "output/dense/kernel": "poswise_networks.{}._layers.2.weight",
        }
        pooler_map = {
            'bert/pooler/dense/bias': 'pooler.0.bias',
            'bert/pooler/dense/kernel': 'pooler.0.weight'
        }
        tf_path = os.path.abspath(os.path.join(cache_dir, 'bert_model.ckpt'))
        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)
        tfnames, arrays = [], []
        for name, _ in init_vars:
            array = tf.train.load_variable(tf_path, name)
            tfnames.append(name)
            arrays.append(array.squeeze())
        py_prefix = "encoder."

        idx = 0
        for name, array in zip(tfnames, arrays):
            if name.startswith('cls'):
                # ignore those variables begin with cls
                continue

            if name in global_tensor_map:
                v_name = global_tensor_map[name]
                pointer = self._name_to_variable(v_name)
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)
                idx += 1
            elif name in pooler_map:
                pointer = self._name_to_variable(pooler_map[name])
                if name.endswith('bias'):
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)
                    idx += 1
                else:
                    array_t = np.transpose(array)
                    assert pointer.shape == array_t.shape
                    pointer.data = torch.from_numpy(array_t)
                    idx += 1
            else:
                # here name is the TensorFlow variable name
                name_tmp = name.split("/")
                # e.g. layer_
                layer_no = name_tmp[2][6:]
                name_tmp = "/".join(name_tmp[3:])
                if name_tmp in layer_tensor_map:
                    v_name = layer_tensor_map[name_tmp].format(layer_no)
                    pointer = self._name_to_variable(py_prefix + v_name)
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)
                elif name_tmp in layer_transpose_map:
                    v_name = layer_transpose_map[name_tmp].format(layer_no)
                    pointer = self._name_to_variable(py_prefix + v_name)
                    array_t = np.transpose(array)
                    assert pointer.shape == array_t.shape
                    pointer.data = torch.from_numpy(array_t)
                else:
                    raise NameError(f"Variable with name '{name}' not found")
                idx += 1
