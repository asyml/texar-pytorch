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
Utils for T5 Modules
"""

import copy
import os
from abc import ABC
from typing import Any, Dict, List, Set

import numpy as np
import torch

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin
from texar.torch.modules.pretrained.t5_utils import read_t5_gin_config_file

__all__ = [
    "PretrainedT5Mixin"
]

_T5_PATH = "https://storage.googleapis.com/t5-data/pretrained_models/"
_T5_VOCAB_PATH = "https://storage.googleapis.com/t5-data/vocabs/cc_all.32000/"
_CHECKPOINT_FILES_GEN_MAP = {  # stores a tuple of model_id and number of
                               # partitions
    'small': (1000000, 16),
    'base': (999900, 16),
    'large': (1000700, 8),
    'B': (1000000, 64)
}


def _generate_t5_file_list(ckpt_tuple: tuple) -> List[str]:
    """ Helper function to generate file list given a tuple of model_id and
    partition size.

    Args:
        ckpt_tuple: A tuple of model_id and number of partitions

    """
    ckpt_id = ckpt_tuple[0]
    ckpt_parts = ckpt_tuple[1]
    return [
        'checkpoint',
        *[f'model.ckpt-{ckpt_id}.data-{idx:05d}-of-{ckpt_parts:05d}'
          for idx in range(ckpt_parts)],
        f'model.ckpt-{ckpt_id}.index',
        f'model.ckpt-{ckpt_id}.meta',
        'operative_config.gin']


class PretrainedT5Mixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the T5 model.

    The T5 model was proposed in
    `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`_
    by `Raffel et al.` from Google. It treats multiple NLP tasks in a similar
    manner by encoding the different tasks as text directives in the input
    stream. This enables a single model to be trained supervised on a wide
    variety of NLP tasks. The T5 model examines factors relevant for leveraging
    transfer learning at scale from pure unsupervised pre-training to
    supervised tasks.

    The available T5 models are as follows:

      * ``T5-Small``: Small version of T5, 60 million parameters.
      * ``T5-Base``: Base-line version of T5, 220 million parameters.
      * ``T5-Large``: Large Version of T5, 770 million parameters.
      * ``T5-3B``: A version of T5 with 3 billion parameters.
      * ``T5-11B``: A version of T5 with 11 billion parameters.

    We provide the following classes:

      * :class:`~texar.torch.modules.T5Encoder` for loading weights for the
        encoder stack.
      * :class:`~texar.torch.modules.T5Decoder` for loading weights for the
        decoding stack.
      * :class:`~texar.torch.modules.T5EncoderDecoder` as a raw pre-trained
        model.

    .. _`Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`:
        https://arxiv.org/abs/1910.10683
    """
    _MODEL_NAME = "T5"

    _MODEL2URL = {
        'T5-Small': [_T5_PATH + f"small/{file}"
                     for file in _generate_t5_file_list(
                      _CHECKPOINT_FILES_GEN_MAP['small'])] +
                    [_T5_VOCAB_PATH + 'sentencepiece.model'],
        'T5-Base': [_T5_PATH + f"base/{file}"
                    for file in _generate_t5_file_list(
                     _CHECKPOINT_FILES_GEN_MAP['base'])] +
                   [_T5_VOCAB_PATH + 'sentencepiece.model'],
        'T5-Large': [_T5_PATH + f"large/{file}"
                     for file in _generate_t5_file_list(
                      _CHECKPOINT_FILES_GEN_MAP['large'])] +
                    [_T5_VOCAB_PATH + 'sentencepiece.model'],
        'T5-3B': [_T5_PATH + f"3B/{file}"
                  for file in _generate_t5_file_list(
                   _CHECKPOINT_FILES_GEN_MAP['B'])] + [_T5_VOCAB_PATH +
                                                       'sentencepiece.model'],
        'T5-11B': [_T5_PATH + f"11B/{file}"
                   for file in _generate_t5_file_list(
                    _CHECKPOINT_FILES_GEN_MAP['B'])] +
                  [_T5_VOCAB_PATH + 'sentencepiece.model']
    }

    _MODEL2CKPT = {
        'T5-Small': 'model.ckpt-1000000',
        'T5-Base': 'model.ckpt-999900',
        'T5-Large': 'model.ckpt-1000700',
        'T5-3B': 'model.ckpt-1000000',
        'T5-11B': 'model.ckpt-1000000'
    }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None
        for file in files:
            if file.endswith('operative_config.gin'):
                config_path = os.path.join(root, file)
        if config_path is None:
            raise ValueError(f"Cannot find the config file in {cache_dir}")

        gin_config = read_t5_gin_config_file(config_path)

        hidden_dim = gin_config['d_model']
        vocab_size = 32128
        eps = 1e-6
        embedding_dropout = gin_config['dropout_rate']
        num_blocks = gin_config['num_layers']
        num_heads = gin_config['num_heads']
        num_units = gin_config['d_kv'] * num_heads
        dropout_rate = gin_config['dropout_rate']
        residual_dropout = gin_config['dropout_rate']
        intermediate_size = gin_config['d_ff']
        rel_attn_num_buckets = 32
        use_bias = False

        configs = {
            'hidden_size': hidden_dim,
            'embed': {
                'name': 'word_embeddings',
                'dim': hidden_dim
            },
            'vocab_size': vocab_size,

            'encoder': {
                'name': 'encoder',
                'embedding_dropout': embedding_dropout,
                'num_blocks': num_blocks,
                'multihead_attention': {
                    'use_bias': use_bias,
                    'num_units': num_units,
                    'num_heads': num_heads,
                    'output_dim': hidden_dim,
                    'dropout_rate': dropout_rate,
                    'name': 'self',
                    'is_decoder': False,
                    'relative_attention_num_buckets': rel_attn_num_buckets
                },
                'eps': eps,
                'residual_dropout': residual_dropout,
                'dim': hidden_dim,
                'poswise_feedforward': {
                    "layers": [{
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': hidden_dim,
                            'out_features': intermediate_size,
                            'bias': use_bias,
                        }
                    }, {
                        'type': "ReLU"
                    }, {
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': intermediate_size,
                            'out_features': hidden_dim,
                            'bias': use_bias,
                        }
                    }],
                },
            },
            'decoder': {
                'name': 'decoder',
                'embedding_dropout': embedding_dropout,
                'num_blocks': num_blocks,
                'multihead_attention': {
                    'use_bias': use_bias,
                    'num_units': num_units,
                    'num_heads': num_heads,
                    'output_dim': hidden_dim,
                    'dropout_rate': dropout_rate,
                    'name': 'self',
                    'is_decoder': True,
                    'relative_attention_num_buckets': rel_attn_num_buckets
                },
                'eps': eps,
                'residual_dropout': residual_dropout,
                'dim': hidden_dim,
                'poswise_feedforward': {
                    "layers": [{
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': hidden_dim,
                            'out_features': intermediate_size,
                            'bias': use_bias,
                        }
                    }, {
                        'type': "ReLU"
                    }, {
                        'type': 'Linear',
                        'kwargs': {
                            'in_features': intermediate_size,
                            'out_features': hidden_dim,
                            'bias': use_bias,
                        }
                    }],
                },
            }

        }

        return configs

    def assign(self, from_array, to_param, transpose=False):
        pointer = self._name_to_variable(to_param)
        if transpose:
            from_array = np.transpose(from_array)
        assert pointer.shape == from_array.shape
        pointer.data = torch.from_numpy(from_array.astype(np.float32))

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str, **kwargs):
        try:
            import tensorflow as tf
        except ImportError:
            print("Loading TensorFlow models in PyTorch requires installing "
                  "TensorFlow. Please see https://www.tensorflow.org/install/ "
                  "for installation instructions.")
            raise

        tf_path = os.path.abspath(os.path.join(
            cache_dir,
            self._MODEL2CKPT[pretrained_model_name]))

        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)

        to_params: Set[str] = {x[0] for x in self.named_parameters()}
        to_params.remove(  # Not used as duplicate weights stored
            'decoder.enc_dec_attns.0.relative_attention_bias.weight')

        tfnames, arrays = [], []
        for name, _ in init_vars:
            array = tf.train.load_variable(tf_path, name)
            tfnames.append(name)
            arrays.append(array.squeeze())

        from_params = set(copy.deepcopy(tfnames))
        global_tensor_map = {
            'shared/embedding': 'word_embedder._embedding'
        }
        self_attention_map = {
            'SelfAttention/k': '{}.self_attns.{}.K_dense.weight',
            'SelfAttention/o': '{}.self_attns.{}.O_dense.weight',
            'SelfAttention/q': '{}.self_attns.{}.Q_dense.weight',
            'SelfAttention/v': '{}.self_attns.{}.V_dense.weight',
            'SelfAttention/relative_attention_bias':
                '{}.self_attns.{}.relative_attention_bias.weight',
            'layer_norm/scale': '{}.self_attn_layer_norm.{}.w'
        }

        enc_dec_attention_map = {
            'EncDecAttention/k': '{}.enc_dec_attns.{}.K_dense.weight',
            'EncDecAttention/o': '{}.enc_dec_attns.{}.O_dense.weight',
            'EncDecAttention/q': '{}.enc_dec_attns.{}.Q_dense.weight',
            'EncDecAttention/v': '{}.enc_dec_attns.{}.V_dense.weight',
            'layer_norm/scale': '{}.end_dec_attn_layer_norm.{}.w'
        }

        drd_map = {
            'DenseReluDense/wi/kernel':
                '{}.poswise_networks.{}._layers.0.weight',
            'DenseReluDense/wo/kernel':
                '{}.poswise_networks.{}._layers.2.weight',
            'layer_norm/scale': '{}.poswise_layer_norm.{}.w'
        }

        component_map = {  # For encoder/decoder level
            'final_layer_norm/scale': '{}.final_layer_norm.w'
        }

        block_map = {
            'encoder0': self_attention_map,
            'encoder1': drd_map,
            'decoder0': self_attention_map,
            'decoder1': enc_dec_attention_map,
            'decoder2': drd_map,
        }

        idx = 0

        # Initialize this param separately
        special_param_name = \
            'decoder.enc_dec_attns.0.relative_attention_bias.weight'
        rab_pointer = self._name_to_variable(special_param_name)
        rab_pointer.data.normal_(mean=0.0,
                                 std=(self._hparams.hidden_size) ** -0.5)

        for name, array in zip(tfnames, arrays):
            if name.startswith('cls') or name == 'global_step' or \
                    name.endswith('adam_m') or name.endswith('adam_v')\
                    or '_slot_' in name:
                # ignore those variables begin with cls
                # ignore 'global_step' variable
                # ignore optimizer state variable
                # ignore slot
                from_params.remove(name)
                continue

            if name in global_tensor_map:
                v_name = global_tensor_map[name]
                self.assign(array, v_name)
                idx += 1
                from_params.remove(name)
                to_params.remove(v_name)
            else:
                # e.g. decoder/block_000/layer_000/SelfAttention/k
                tmp_name = name.split('/')
                submodule = tmp_name[0]  # encoder/decoder
                if len(tmp_name) > 3:  # has block level data
                    block_num = int(tmp_name[1][6:])
                    layer_num = str(int(tmp_name[2][6:]))
                    sublayer_name = "/".join(tmp_name[3:])
                    # Block-wise params
                    map_ = block_map[submodule + layer_num]
                    if sublayer_name in map_:
                        v_name = map_[sublayer_name].format(submodule,
                                                            block_num)
                        self.assign(array, v_name, True)
                        idx += 1
                        from_params.remove(name)
                        to_params.remove(v_name)
                else:
                    # e.g. decoder/final_layer_norm/scale
                    sublayer_name = "/".join(tmp_name[1:])
                    if sublayer_name in component_map:
                        v_name = component_map[sublayer_name].format(submodule)
                        self.assign(array, v_name)
                        idx += 1
                        from_params.remove(name)
                        to_params.remove(v_name)

        if len(from_params) > 0:
            print(f"WARNING: Certain weights from checkpoint are not loaded: "
                  f"{list(from_params)}")

        if len(to_params) > 0:
            print(f"WARNING: Certain parameters are not initialized: "
                  f"{list(to_params)}")
