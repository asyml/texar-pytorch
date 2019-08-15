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
_ROBERTA_PATH = "https://dl.fbaipublicfiles.com/fairseq/models/"


class PretrainedBERTMixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the BERT model.

    Both standard BERT models and many variants are supported. Usually, you can
    specify the :attr:`pretrained_model_name` argument to pick which pre-trained
    BERT model to use. All available categories of pre-trained models
    (and names) include:

      * **Standard BERT**: proposed in (`Devlin et al`. 2018)
        `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
        . A bidirectional Transformer language model pre-trained on large text
        corpora. Available model names include:

        * ``bert-base-uncased``: 12-layer, 768-hidden, 12-heads,
          110M parameters.
        * ``bert-large-uncased``: 24-layer, 1024-hidden, 16-heads,
          340M parameters.
        * ``bert-base-cased``: 12-layer, 768-hidden, 12-heads , 110M parameters.
        * ``bert-large-cased``: 24-layer, 1024-hidden, 16-heads,
          340M parameters.
        * ``bert-base-multilingual-uncased``: 102 languages, 12-layer,
          768-hidden, 12-heads, 110M parameters.
        * ``bert-base-multilingual-cased``: 104 languages, 12-layer, 768-hidden,
          12-heads, 110M parameters.
        * ``bert-base-chinese``: Chinese Simplified and Traditional, 12-layer,
          768-hidden, 12-heads, 110M parameters.

      * **RoBERTa**: proposed in (`Liu et al`. 2019)
        `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
        . As a variant of the standard BERT model, RoBERTa trains for more
        iterations on more data with a larger batch size as well as other tweaks
        in pre-training. Differing from the standard BERT, the RoBERTa model
        does not use segmentation embedding. Available model names include:

        * ``roberta-base``: RoBERTa using the BERT-base architecture,
          125M parameters.
        * ``roberta-large``: RoBERTa using the BERT-large architecture,
          355M parameters.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692
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

        if pretrained_model_name.startswith('bert'):
            for file in files:
                if file.endswith('config.json'):
                    config_path = os.path.join(root, file)
                    with open(config_path) as f:
                        config_ckpt = json.loads(f.read())
                        hidden_dim = config_ckpt['hidden_size']
                        vocab_size = config_ckpt['vocab_size']
                        type_vocab_size = config_ckpt['type_vocab_size']
                        position_size = config_ckpt['max_position_embeddings']
                        embedding_dropout = config_ckpt['hidden_dropout_prob']
                        num_blocks = config_ckpt['num_hidden_layers']
                        num_heads = config_ckpt['num_attention_heads']
                        dropout_rate = config_ckpt[
                            'attention_probs_dropout_prob']
                        residual_dropout = config_ckpt['hidden_dropout_prob']
                        intermediate_size = config_ckpt['intermediate_size']
                        hidden_act = config_ckpt['hidden_act']

        elif pretrained_model_name.startswith('roberta'):
            for file in files:
                if file.endswith('model.pt'):
                    config_path = os.path.join(root, file)
                    args = torch.load(config_path, map_location="cpu")['args']
                    hidden_dim = args.encoder_embed_dim
                    vocab_size = 50265
                    type_vocab_size = 0
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
            'segment_embed': {
                'name': 'token_type_embeddings',
                'dim': hidden_dim
            },
            'type_vocab_size': type_vocab_size,
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
        if pretrained_model_name.startswith('bert'):
            self._init_bert_from_checkpoint(cache_dir)
        elif pretrained_model_name.startswith('roberta'):
            self._init_roberta_from_checkpoint(cache_dir)

    def _init_bert_from_checkpoint(self, cache_dir: str):
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

    def _init_roberta_from_checkpoint(self, cache_dir: str):
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
                layer_num, layer_name = name[0], name[2:]
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
