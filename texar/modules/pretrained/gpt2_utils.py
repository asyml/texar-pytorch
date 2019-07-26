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

from typing import Dict, Optional

import json
import os
import sys

import torch
import torch.nn as nn

from texar.core import layers
from texar.data.data_utils import maybe_download
from texar.modules.pretrained.pretrained_utils import default_download_dir


__all__ = [
    "init_gpt2_checkpoint",
    "load_pretrained_gpt2",
    "transform_gpt2_to_texar_config",
]


_GPT2_PATH = "https://storage.googleapis.com/gpt-2/models/"
_MODEL2URL = {
    '117M': _GPT2_PATH + "117M",
    '345M': _GPT2_PATH + "345M",
}


def init_gpt2_checkpoint(model: nn.Module, cache_dir: str):
    r"""Initializes GPT2 model parameters from a checkpoint provided by Google.
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to "
              "be installed. Please see https://www.tensorflow.org/install/ "
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
    for name, _ in model.word_embedder.named_parameters():
        tensor_names.append(name)
    for name, _ in model.position_embedder.named_parameters():
        tensor_names.append(name)
    for name, _ in model.decoder.named_parameters():
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
                pointer = model.word_embedder.embedding
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)

                output_pointer = name_to_variable(
                    model.decoder, "_output_layer.weight")
                if (output_pointer is not layers.identity) or \
                        (not isinstance(output_pointer, layers.Identity)):
                    assert output_pointer.shape == array.shape
                    output_pointer.data = torch.from_numpy(array)
            elif name == "model/wpe":
                pointer = model.position_embedder.embedding
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)
            else:
                pointer = name_to_variable(model.decoder, v_name)
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)

        else:
            name_tmp = name.split("/")
            layer_no = name_tmp[1][1:]
            name = "/".join(name_tmp[2:])
            if name in layer_tensor_map:
                v_name = layer_tensor_map[name].format(layer_no)
                pointer = name_to_variable(model.decoder, v_name)
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)
            elif name in layer_transpose_map:
                v_name = layer_transpose_map[name].format(layer_no)
                pointer = name_to_variable(model.decoder, v_name)
                array_t = np.transpose(array)
                assert pointer.shape == array_t.shape
                pointer.data = torch.from_numpy(array_t)
            elif name == "attn/c_attn/w":
                index_d = array.shape[-1] // 3

                Q_w = np.transpose(array[:, :index_d])
                K_w = np.transpose(array[:, index_d: 2 * index_d])
                V_w = np.transpose(array[:, 2 * index_d:])

                q_weight = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.Q_dense.weight")
                k_weight = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.K_dense.weight")
                v_weight = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.V_dense.weight")

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
                q_bias = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.Q_dense.bias")
                k_bias = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.K_dense.bias")
                v_bias = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.V_dense.bias")

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
            if (pointer is not layers.identity) or \
                    (not isinstance(pointer, layers.Identity)):
                pointer = getattr(pointer, m_name)
    return pointer


def load_pretrained_gpt2(pretrained_model_name: str,
                         cache_dir: Optional[str] = None) -> str:
    r"""Return the directory in which the pretrained `GPT2` is cached.
    """
    if pretrained_model_name in _MODEL2URL:
        download_path = _MODEL2URL[pretrained_model_name]
    else:
        raise ValueError(
            "Pre-trained model not found: {}".format(pretrained_model_name))

    if cache_dir is None:
        cache_dir = default_download_dir("gpt2")

    file_name = download_path.split('/')[-1]

    cache_path = os.path.join(cache_dir, file_name.split('.')[0])
    if not os.path.exists(cache_path):
        for filename in ["checkpoint", "encoder.json", "hparams.json",
                         "model.ckpt.data-00000-of-00001", "model.ckpt.index",
                         "model.ckpt.meta", "vocab.bpe"]:
            maybe_download(os.path.join(download_path, filename), cache_path)
    else:
        print("Using cached pre-trained GPT2 model from: %s." % cache_path)

    return cache_path


def transform_gpt2_to_texar_config(cache_dir: str) -> Dict:
    r"""Load the Json config file and transform it into Texar style
    configuration.
    """
    info = list(os.walk(cache_dir))
    root, _, files = info[0]
    config_path = None
    for file in files:
        if file.endswith('hparams.json'):
            config_path = os.path.join(root, file)
    if config_path is None:
        raise ValueError("Cannot find the config file in {}".format(cache_dir))

    with open(config_path) as f:
        config_gpt = json.loads(f.read())

    configs = {}
    configs["vocab_size"] = config_gpt["n_vocab"]
    configs["context_size"] = config_gpt["n_ctx"]
    configs["embedding_size"] = config_gpt["n_embd"]
    hidden_dim = config_gpt["n_embd"]
    configs["embed"] = {
        "dim": hidden_dim,
    }
    configs["position_size"] = config_gpt["n_ctx"]
    configs["position_embed"] = {
        "dim": hidden_dim
    }
    configs["decoder"] = {
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
    }
    return configs
