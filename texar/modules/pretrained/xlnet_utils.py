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
Utils of XLNet Modules.
"""

from typing import Callable, Dict, List, Optional, Union

import json
import os
import sys
import numpy as np

import torch
import torch.nn as nn

from texar.data.data_utils import maybe_download
from texar.modules.pretrained.xlnet_model_utils import \
    (PositionWiseFF, RelativeMultiheadAttention)


__all__ = [
    "init_xlnet_checkpoint",
    "load_pretrained_model",
    "transform_xlnet_to_texar_config",
    "sum_tensors",
]


_XLNET_PATH = "https://storage.googleapis.com/xlnet/released_models/"
_MODEL2URL = {
    'xlnet-base-cased':
        _XLNET_PATH + "cased_L-12_H-768_A-12.zip",
    'xlnet-large-cased':
        _XLNET_PATH + "cased_L-24_H-1024_A-16.zip",
}


def init_xlnet_checkpoint(model: nn.Module, cache_dir: str):
    r"""Initializes XLNet model parameters from a checkpoint provided by Google.
    """
    # remember to call .contiguous after trans_fn
    import tensorflow as tf
    ckpt = tf.train.load_checkpoint(os.path.join(cache_dir, 'xlnet_model.ckpt'))
    from_params: Dict[str, np.ndarray] = {
        key: ckpt.get_tensor(key)
        for key in ckpt.get_variable_to_shape_map().keys()}
    del from_params["global_step"]  # useless variable
    to_params: Dict[str, nn.Parameter] = dict(model.named_parameters())

    def get_weight(name: str) -> torch.Tensor:
        weight = from_params["model/" + name]
        del from_params["model/" + name]
        return torch.from_numpy(weight)

    TransFn = Callable[[torch.Tensor], torch.Tensor]

    def assign(param: nn.Parameter, weight: Union[str, torch.Tensor],
               trans_fn: Optional[TransFn] = None, allow_fail: bool = False):
        param_key = next(k for k, v in to_params.items() if v is param)
        del to_params[param_key]  # delete regardless of whether weight exists
        if isinstance(weight, str):
            try:
                weight = get_weight(weight)
            except KeyError:
                if allow_fail:
                    print(f"Weight {weight} not found in checkpoint")
                    return
                else:
                    raise
        if trans_fn is not None:
            weight = trans_fn(weight).contiguous()
        if param.size() != weight.size():
            raise ValueError(f"Expected size {param.size()}, "
                             f"actual size {weight.size()}")
        param.data = weight

    def assign_linear(linear: nn.Linear, prefix: str):
        trans_fn = lambda p: p.view(p.size(0), -1).t()
        assign(linear.weight, prefix + "kernel", trans_fn)
        if linear.bias is not None:
            assign(linear.bias, prefix + "bias")

    def assign_layer_norm(layer_norm: nn.LayerNorm, prefix: str):
        assign(layer_norm.weight, prefix + "LayerNorm/gamma")
        assign(layer_norm.bias, prefix + "LayerNorm/beta")

    def load_xlnet_model(xlnet):
        n_layers = len(xlnet.attn_layers)
        for bias_name in ['r_r_bias', 'r_w_bias', 'r_s_bias']:
            weight = get_weight("transformer/" + bias_name)
            if xlnet.hparams.untie_r:
                for idx in range(n_layers):
                    layer: RelativeMultiheadAttention = xlnet.attn_layers[idx]
                    assign(getattr(layer, bias_name), weight[idx])
            else:
                assign(getattr(xlnet, bias_name), weight)
        assign(xlnet.word_embed.weight,
               "transformer/word_embedding/lookup_table")

        for idx in range(n_layers):
            layer: RelativeMultiheadAttention = xlnet.attn_layers[idx]
            prefix = f"transformer/layer_{idx}/rel_attn/"
            qkv_weights = [get_weight(prefix + f"{part}/kernel")
                           for part in "qkv"]
            assign(layer.head_projection.weight,
                   torch.cat([
                       p.view(p.size(0), -1) for p in qkv_weights
                   ], dim=1).t())
            assign_linear(layer.pos_projection, prefix + "r/")
            assign(layer.output_projection.weight,  # DO NOT TRANSPOSE THIS!!!!
                   prefix + "o/kernel", lambda p: p.view(p.size(0), -1))
            assign_layer_norm(layer.layer_norm, prefix)

        for idx in range(n_layers):
            layer: PositionWiseFF = xlnet.ff_layers[idx]
            prefix = f"transformer/layer_{idx}/ff/"
            for linear_idx in range(1, 2 + 1):
                linear_prefix = f"{prefix}layer_{linear_idx}/"
                linear_layer: nn.Linear = getattr(layer, f"linear{linear_idx}")
                assign_linear(linear_layer, linear_prefix)
            assign_layer_norm(layer.layer_norm, prefix)

        seg_embeds = [
            p.squeeze(0)
            for p in torch.chunk(
                get_weight("transformer/seg_embed"), n_layers, dim=0)]
        for idx in range(n_layers):
            assign(xlnet.attn_layers[idx].segment_embed, seg_embeds[idx])

        if isinstance(xlnet, XLNetDecoder):
            assign(xlnet.mask_emb, "transformer/mask_emb/mask_emb")
            assign(xlnet.lm_bias, "lm_loss/bias")

    from texar.modules.encoders.xlnet_encoder import XLNetEncoder
    from texar.modules.decoders.xlnet_decoder import XLNetDecoder

    if isinstance(model, XLNetEncoder) or isinstance(model, XLNetDecoder):
        load_xlnet_model(model)
    else:
        raise ValueError("The specified model must be an XLNet model.")

    if len(from_params) > 0:
        print(f"WARNING: Certain weights from checkpoint are not loaded: "
              f"{list(from_params.keys())}")

    filtered_to_params = [k for k in to_params if k.startswith("xlnet")]
    if len(filtered_to_params) > 0:
        print(f"WARNING: Certain parameters are not initialized: "
              f"{list(filtered_to_params)}")


def _default_download_dir() -> str:
    r"""Return the directory to which packages will be downloaded by default.
    """
    package_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))
    if os.access(package_dir, os.W_OK):
        texar_download_dir = os.path.join(package_dir, 'texar_download')
    else:
        # On Windows, use %APPDATA%
        if sys.platform == 'win32' and 'APPDATA' in os.environ:
            home_dir = os.environ['APPDATA']

        # Otherwise, install in the user's home directory.
        else:
            home_dir = os.path.expanduser('~/')
            if home_dir == '~/':
                raise ValueError("Could not find a default download directory")

        texar_download_dir = os.path.join(home_dir, 'texar_download')

    if not os.path.exists(texar_download_dir):
        os.mkdir(texar_download_dir)

    return os.path.join(texar_download_dir, 'xlnet')


def load_pretrained_model(pretrained_model_name: str,
                          cache_dir: Optional[str] = None) -> str:
    r"""Return the directory in which the pretrained model is cached.
    """
    if pretrained_model_name in _MODEL2URL:
        download_path = _MODEL2URL[pretrained_model_name]
    else:
        raise ValueError(
            "Pre-trained model not found: {}".format(pretrained_model_name))

    if cache_dir is None:
        cache_dir = _default_download_dir()

    file_name = download_path.split('/')[-1]

    cache_path = os.path.join(cache_dir, 'xlnet_' + file_name.split('.')[0])
    if not os.path.exists(cache_path):
        maybe_download(download_path, cache_dir, extract=True)
    else:
        print("Using cached pre-trained XLNet model from: %s." % cache_path)

    return cache_path


def transform_xlnet_to_texar_config(cache_dir: str) -> Dict:
    r"""Load the Json config file and transform it into Texar style
    configuration.
    """
    info = list(os.walk(cache_dir))
    root, _, files = info[0]
    config_path = None
    for file in files:
        if file.endswith('config.json'):
            config_path = os.path.join(root, file)
    if config_path is None:
        raise ValueError("Cannot find the config file in {}".format(cache_dir))

    with open(config_path) as f:
        config_ckpt = json.loads(f.read())

    configs = {}
    configs["head_dim"] = config_ckpt["d_head"]
    configs["ffn_inner_dim"] = config_ckpt["d_inner"]
    configs["hidden_dim"] = config_ckpt["d_model"]
    configs["activation"] = config_ckpt["ff_activation"]
    configs["num_heads"] = config_ckpt["n_head"]
    configs["num_layers"] = config_ckpt["n_layer"]
    configs["vocab_size"] = config_ckpt["n_token"]
    configs["untie_r"] = config_ckpt["untie_r"]

    return configs


def sum_tensors(xs: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    r"""Sum a list of tensors with possible `None` values.
    """
    idx = next((idx for idx, tensor in enumerate(xs) if tensor is not None), -1)
    if idx == -1:
        return None
    ret = xs[idx]
    for tensor in xs[(idx + 1):]:
        if tensor is not None:
            ret = ret + tensor
    return ret
