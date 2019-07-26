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
import sys
from typing import Any, Dict, Optional

import torch
from torch import nn

from texar.data.data_utils import maybe_download
from texar.modules.pretrained.pretrained_base import PretrainedMixin
from texar.modules.pretrained.pretrained_utils import default_download_dir

__all__ = [
    "PretrainedGPT2Mixin",
]

_GPT2_PATH = "https://storage.googleapis.com/gpt-2/models/"
_MODEL2URL = {
    '117M': _GPT2_PATH + "117M",
    '345M': _GPT2_PATH + "345M",
}


class PretrainedGPT2Mixin(PretrainedMixin):
    @staticmethod
    def _download_checkpoint(pretrained_model_name: str,
                             cache_dir: Optional[str] = None) -> str:
        r"""Return the directory in which the pretrained `GPT2` is cached.
            """
        if pretrained_model_name in _MODEL2URL:
            download_path = _MODEL2URL[pretrained_model_name]
        else:
            raise ValueError(
                f"Pre-trained model not found: {pretrained_model_name}")

        if cache_dir is None:
            cache_dir = default_download_dir("gpt2")

        file_name = download_path.split('/')[-1]

        cache_path = os.path.join(cache_dir, file_name.split('.')[0])
        if not os.path.exists(cache_path):
            files_to_load = [
                "checkpoint", "encoder.json", "hparams.json",
                "model.ckpt.data-00000-of-00001", "model.ckpt.index",
                "model.ckpt.meta", "vocab.bpe"]
            for filename in files_to_load:
                maybe_download(
                    os.path.join(download_path, filename), cache_path)
        else:
            print(f"Using cached pre-trained GPT2 model from: {cache_path}.")

        return cache_path

    @staticmethod
    def _transform_config(cache_dir: str) -> Dict[str, Any]:
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
        configs.update({
            "dim": hidden_dim,
            "num_blocks": config_gpt["n_layer"],
            "use_gpt_config": True,
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
        })
        return configs

    def _init_from_checkpoint(self, cache_dir: str,
                              load_output_layer: bool = True, **kwargs):
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            print("Loading TensorFlow models in PyTorch requires installing "
                  "TensorFlow. Please see https://www.tensorflow.org/install/ "
                  "for installation instructions.")
            raise

        global_tensor_map = {
            "model/wte": "_embedding",
            "model/wpe": "_embedding",
            "model/ln_f/b": "final_layer_norm.bias",
            "model/ln_f/g": "final_layer_norm.weight",
        }
        layer_tensor_map = {
            "ln_1/b": "self_attn_layer_norm.{}.bias",
            "ln_1/g": "self_attn_layer_norm.{}.weight",
            "ln_2/b": "poswise_layer_norm.{}.bias",
            "ln_2/g": "poswise_layer_norm.{}.weight",
            "mlp/c_fc/b": "poswise_networks.{}._layers.0.bias",
            "mlp/c_proj/b": "poswise_networks.{}._layers.2.bias",
            "attn/c_proj/b": "self_attns.{}.O_dense.bias",
        }
        layer_transpose_map = {
            "mlp/c_fc/w": "poswise_networks.{}._layers.0.weight",
            "mlp/c_proj/w": "poswise_networks.{}._layers.2.weight",
            "attn/c_proj/w": "self_attns.{}.O_dense.weight",
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
        for name, _ in self.word_embedder.named_parameters():
            tensor_names.append(name)
        for name, _ in self.position_embedder.named_parameters():
            tensor_names.append(name)
        for name, _ in self.named_parameters():
            tensor_names.append(name)

        idx = 0
        for name, array in zip(names, arrays):

            processing = (idx + 1.0) / len(names)
            idx += 1
            sys.stdout.write(f"\rLoading checkpoint: {processing:.1%}")
            sys.stdout.flush()

            if name in global_tensor_map:
                v_name = global_tensor_map[name]

                if name == "model/wte":
                    if load_output_layer:
                        pointer = self.word_embedder.embedding
                        assert pointer.shape == array.shape
                        pointer.data = torch.from_numpy(array)

                        output_pointer = name_to_variable(
                            self, "_output_layer.weight")
                        if not isinstance(output_pointer, nn.Identity):
                            assert output_pointer.shape == array.shape
                            output_pointer.data = torch.from_numpy(array)
                elif name == "model/wpe":
                    pointer = self.position_embedder.embedding
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
                        f"self_attns.{layer_no}.Q_dense.weight")
                    k_weight = self._name_to_variable(
                        f"self_attns.{layer_no}.K_dense.weight")
                    v_weight = self._name_to_variable(
                        f"self_attns.{layer_no}.V_dense.weight")

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
                        f"self_attns.{layer_no}.Q_dense.bias")
                    k_bias = self._name_to_variable(
                        f"self_attns.{layer_no}.K_dense.bias")
                    v_bias = self._name_to_variable(
                        f"self_attns.{layer_no}.V_dense.bias")

                    assert q_bias.shape == Q_b.shape
                    assert k_bias.shape == K_b.shape
                    assert v_bias.shape == V_b.shape

                    q_bias.data = torch.from_numpy(Q_b)
                    k_bias.data = torch.from_numpy(K_b)
                    v_bias.data = torch.from_numpy(V_b)

                else:
                    print("Name error", name)
                    raise Exception


def name_to_variable(model: nn.Module, name: str) -> nn.Module:
    r"""Find the corresponding variable given the specified name.
    """
    pointer = model
    name = name.split(".")
    for m_name in name:
        if m_name.isdigit():
            num = int(m_name)
            pointer = pointer[num]  # type: ignore
        else:
            if not isinstance(pointer, nn.Identity):
                pointer = getattr(pointer, m_name)
    return pointer
