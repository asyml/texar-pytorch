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
Utils of GPT2 Modules.
"""

import json
import os
import warnings
from abc import ABC
from typing import Any, Dict

import torch

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedGPT2Mixin",
]

_GPT2_PATH = "https://storage.googleapis.com/gpt-2/models/"
_CHECKPOINT_FILES = [
    "checkpoint", "encoder.json", "hparams.json", "vocab.bpe",
    "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta"]


class PretrainedGPT2Mixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the GPT2 model.

    The GPT2 model was proposed in
    `Language Models are Unsupervised Multitask Learners`_
    by `Radford et al.` from OpenAI. It is a unidirectional Transformer model
    pre-trained using the vanilla language modeling objective on a large corpus.

    The available GPT2 models are as follows:

      * ``gpt2-small``: Small version of GPT-2, 124M parameters.
      * ``gpt2-medium``: Medium version of GPT-2, 355M parameters.
      * ``gpt2-large``: Large version of GPT-2, 774M parameters.
      * ``gpt2-xl``: XL version of GPT-2, 1558M parameters.

    We provide the following GPT2 classes:

      * :class:`~texar.torch.modules.GPT2Encoder` for text encoding.
      * :class:`~texar.torch.modules.GPT2Decoder` for text generation and
        decoding.
      * :class:`~texar.torch.modules.GPT2Classifier` for text classification and
        sequence tagging.

    .. _`Language Models are Unsupervised Multitask Learners`:
        https://openai.com/blog/better-language-models/
    """
    _MODEL_NAME = "GPT2"
    _MODEL2URL = {
        'gpt2-small': [_GPT2_PATH + f"124M/{file}"
                       for file in _CHECKPOINT_FILES],
        'gpt2-medium': [_GPT2_PATH + f"355M/{file}"
                        for file in _CHECKPOINT_FILES],
        'gpt2-large': [_GPT2_PATH + f"774M/{file}"
                       for file in _CHECKPOINT_FILES],
        'gpt2-xl': [_GPT2_PATH + f"1558M/{file}"
                    for file in _CHECKPOINT_FILES],
    }

    _IS_DECODE = False

    # Raise warning for the deprecated pre-trained model names
    class MyDict(dict):
        def __contains__(self, key):
            if key == '117M':
                warnings.warn("Pre-trained model name '117M' is deprecated, "
                              "use 'gpt2-small' instead.", UserWarning)
                return True
            elif key == '345M':
                warnings.warn("Pre-trained model name '345M' is deprecated, "
                              "use 'gpt2-medium' instead.", UserWarning)
                return True
            else:
                return super().__contains__(key)
    _DEPRECATED_MODEL2URL = {
        '117M': [_GPT2_PATH + f"124M/{file}" for file in _CHECKPOINT_FILES],
        '345M': [_GPT2_PATH + f"355M/{file}" for file in _CHECKPOINT_FILES],
    }
    _MODEL2URL.update(_DEPRECATED_MODEL2URL)
    _MODEL2URL = MyDict(_MODEL2URL)  # type: ignore

    def _transform_config(self, pretrained_model_name: str,  # type: ignore
                          cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None
        for file in files:
            if file.endswith('hparams.json'):
                config_path = os.path.join(root, file)
        if config_path is None:
            raise ValueError(f"Cannot find the config file in {cache_dir}")

        with open(config_path) as f:
            config_gpt = json.loads(f.read())

        hidden_dim = config_gpt["n_embd"]
        configs = {
            "vocab_size": config_gpt["n_vocab"],
            "context_size": config_gpt["n_ctx"],
            "embedding_size": config_gpt["n_embd"], "embed": {
                "dim": hidden_dim,
            },
            "position_size": config_gpt["n_ctx"],
            "position_embed": {
                "dim": hidden_dim
            }
        }

        module_name = 'decoder' if self._IS_DECODE else 'encoder'
        configs.update({module_name: {
            "dim": hidden_dim,
            "num_blocks": config_gpt["n_layer"],
            "embedding_dropout": 0,
            "residual_dropout": 0,
            "multihead_attention": {
                "use_bias": True,
                "num_units": hidden_dim,
                "num_heads": config_gpt["n_head"],
                "output_dim": hidden_dim,
            },
            "initializer": {
                "type": "variance_scaling_initializer",
                "kwargs": {
                    "factor": 1.0,
                    "mode": "FAN_AVG",
                    "uniform": True,
                },
            },
            "poswise_feedforward": {
                "layers": [
                    {
                        "type": "Linear",
                        "kwargs": {
                            "in_features": hidden_dim,
                            "out_features": hidden_dim * 4,
                            "bias": True,
                        }
                    },
                    {
                        "type": "GPTGELU",
                        "kwargs": {}
                    },
                    {
                        "type": "Linear",
                        "kwargs": {
                            "in_features": hidden_dim * 4,
                            "out_features": hidden_dim,
                            "bias": True,
                        }
                    }
                ],
                "name": "ffn",
            },
        }})
        if self._IS_DECODE:
            configs[module_name].update({'use_gpt_config': True})
        else:
            configs[module_name].update({'use_bert_config': False})
        return configs

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str,
                              load_output_layer: bool = True, **kwargs):
        r"""Initialize model parameters from weights stored in the pre-trained
        checkpoint.

        Args:
            pretrained_model_name (str): Name of the pre-trained model.
            cache_dir (str): Path to the cache directory.
            load_output_layer (bool): If `False`, will not load weights of the
                output layer. Set this argument to `False` when loading weights
                into a GPT2 encoder. Defaults to `True`.
        """
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            print("Loading TensorFlow models in PyTorch requires installing "
                  "TensorFlow. Please see https://www.tensorflow.org/install/ "
                  "for installation instructions.")
            raise

        module_name = 'decoder' if self._IS_DECODE else 'encoder'

        global_tensor_map = {
            "model/wte": "word_embedder.embedding",
            "model/wpe": "position_embedder.embedding",
            "model/ln_f/b": module_name + ".final_layer_norm.bias",
            "model/ln_f/g": module_name + ".final_layer_norm.weight",
        }
        layer_tensor_map = {
            "ln_1/b": module_name + ".self_attn_layer_norm.{}.bias",
            "ln_1/g": module_name + ".self_attn_layer_norm.{}.weight",
            "ln_2/b": module_name + ".poswise_layer_norm.{}.bias",
            "ln_2/g": module_name + ".poswise_layer_norm.{}.weight",
            "mlp/c_fc/b": module_name + ".poswise_networks.{}._layers.0.bias",
            "mlp/c_proj/b": module_name + ".poswise_networks.{}._layers.2.bias",
            "attn/c_proj/b": module_name + ".self_attns.{}.O_dense.bias",
        }
        layer_transpose_map = {
            "mlp/c_fc/w": module_name + ".poswise_networks.{}._layers.0.weight",
            "mlp/c_proj/w": module_name + ".poswise_networks.{}._layers.2."
                                          "weight",
            "attn/c_proj/w": module_name + ".self_attns.{}.O_dense.weight",
        }

        tf_path = os.path.abspath(os.path.join(cache_dir, 'model.ckpt'))
        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)
        names = []
        arrays = []
        for name, _ in init_vars:
            array = tf.train.load_variable(tf_path, name)
            names.append(name)
            arrays.append(array.squeeze())

        tensor_names = []
        for name, _ in self.named_parameters():
            tensor_names.append(name)

        for name, array in zip(names, arrays):
            if name in global_tensor_map:
                v_name = global_tensor_map[name]
                if name == "model/wte":
                    pointer = self._name_to_variable(v_name)
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)

                    if load_output_layer:
                        output_pointer = self._name_to_variable(
                            "decoder._output_layer.weight")
                        assert output_pointer.shape == array.shape
                        output_pointer.data = torch.from_numpy(array)
                elif name == "model/wpe":
                    pointer = self._name_to_variable(v_name)
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)
                else:
                    pointer = self._name_to_variable(v_name)
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)

            else:
                name_tmp = name.split("/")
                layer_no = name_tmp[1][1:]
                name = "/".join(name_tmp[2:])
                if name in layer_tensor_map:
                    v_name = layer_tensor_map[name].format(layer_no)
                    pointer = self._name_to_variable(v_name)
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)
                elif name in layer_transpose_map:
                    v_name = layer_transpose_map[name].format(layer_no)
                    pointer = self._name_to_variable(v_name)
                    array_t = np.transpose(array)
                    assert pointer.shape == array_t.shape
                    pointer.data = torch.from_numpy(array_t)
                elif name == "attn/c_attn/w":
                    index_d = array.shape[-1] // 3

                    Q_w = np.transpose(array[:, :index_d])
                    K_w = np.transpose(array[:, index_d: 2 * index_d])
                    V_w = np.transpose(array[:, 2 * index_d:])

                    q_weight = self._name_to_variable(
                        f"{module_name}.self_attns.{layer_no}.Q_dense.weight")
                    k_weight = self._name_to_variable(
                        f"{module_name}.self_attns.{layer_no}.K_dense.weight")
                    v_weight = self._name_to_variable(
                        f"{module_name}.self_attns.{layer_no}.V_dense.weight")

                    assert q_weight.shape == Q_w.shape
                    assert k_weight.shape == K_w.shape
                    assert v_weight.shape == V_w.shape

                    q_weight.data = torch.from_numpy(Q_w)
                    k_weight.data = torch.from_numpy(K_w)
                    v_weight.data = torch.from_numpy(V_w)

                elif name == "attn/c_attn/b":
                    d = array.shape[0]
                    Q_b = array[: d // 3]
                    K_b = array[d // 3: 2 * d // 3]
                    V_b = array[2 * d // 3:]
                    q_bias = self._name_to_variable(
                        f"{module_name}.self_attns.{layer_no}.Q_dense.bias")
                    k_bias = self._name_to_variable(
                        f"{module_name}.self_attns.{layer_no}.K_dense.bias")
                    v_bias = self._name_to_variable(
                        f"{module_name}.self_attns.{layer_no}.V_dense.bias")

                    assert q_bias.shape == Q_b.shape
                    assert k_bias.shape == K_b.shape
                    assert v_bias.shape == V_b.shape

                    q_bias.data = torch.from_numpy(Q_b)
                    k_bias.data = torch.from_numpy(K_b)
                    v_bias.data = torch.from_numpy(V_b)

                else:
                    print("Name error", name)
                    raise Exception
