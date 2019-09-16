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

import json
import os
from abc import ABC
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin
from texar.torch.modules.pretrained.xlnet_utils import (
    PositionWiseFF, RelativeMultiheadAttention, init_weights)

__all__ = [
    "PretrainedXLNetMixin",
]

_XLNET_PATH = "https://storage.googleapis.com/xlnet/released_models/"


class PretrainedXLNetMixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the XLNet model.

    The XLNet model was proposed in
    `XLNet: Generalized Autoregressive Pretraining for Language Understanding`_
    by `Yang et al.` It is based on the Transformer-XL model, pre-trained on a
    large corpus using a language modeling objective that considers all
    permutations of the input sentence.

    The available XLNet models are as follows:

      * ``xlnet-based-cased``: 12-layer, 768-hidden, 12-heads. This model is
        trained on full data (different from the one in the paper).
      * ``xlnet-large-cased``: 24-layer, 1024-hidden, 16-heads.

    We provide the following XLNet classes:

      * :class:`~texar.torch.modules.XLNetEncoder` for text encoding.
      * :class:`~texar.torch.modules.XLNetDecoder` for text generation and
        decoding.
      * :class:`~texar.torch.modules.XLNetClassifier` for text classification
        and sequence tagging.
      * :class:`~texar.torch.modules.XLNetRegressor` for text regression.

    .. _`XLNet: Generalized Autoregressive Pretraining for Language Understanding`:
        http://arxiv.org/abs/1906.08237
    """
    _MODEL_NAME = "XLNet"
    _MODEL2URL = {
        'xlnet-base-cased':
            _XLNET_PATH + "cased_L-12_H-768_A-12.zip",
        'xlnet-large-cased':
            _XLNET_PATH + "cased_L-24_H-1024_A-16.zip",
    }

    def reset_parameters(self):
        self.apply(init_weights)
        if not self._hparams.untie_r:
            nn.init.normal_(self.r_w_bias, 0.0, 0.02)
            nn.init.normal_(self.r_r_bias, 0.0, 0.02)
            if self._hparams.use_segments:
                nn.init.normal_(self.r_s_bias, 0.0, 0.02)

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
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

        configs = {
            "head_dim": config_ckpt["d_head"],
            "ffn_inner_dim": config_ckpt["d_inner"],
            "hidden_dim": config_ckpt["d_model"],
            "activation": config_ckpt["ff_activation"],
            "num_heads": config_ckpt["n_head"],
            "num_layers": config_ckpt["n_layer"],
            "vocab_size": config_ckpt["n_token"],
            "untie_r": config_ckpt["untie_r"]
        }

        return configs

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str, **kwargs):
        # remember to call .contiguous after trans_fn
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            print("Loading TensorFlow models in PyTorch requires installing "
                  "TensorFlow. Please see https://www.tensorflow.org/install/ "
                  "for installation instructions.")
            raise

        ckpt = tf.train.load_checkpoint(
            os.path.join(cache_dir, 'xlnet_model.ckpt'))
        from_params: Dict[str, np.ndarray] = {
            key: ckpt.get_tensor(key)
            for key in ckpt.get_variable_to_shape_map().keys()}
        del from_params["global_step"]  # useless variable
        to_params: Dict[str, nn.Parameter] = dict(self.named_parameters())

        def get_weight(name: str) -> torch.Tensor:
            weight = from_params["model/" + name]
            del from_params["model/" + name]
            return torch.from_numpy(weight)

        TransFn = Callable[[torch.Tensor], torch.Tensor]

        def assign(param: nn.Parameter, weight: Union[str, torch.Tensor],
                   trans_fn: Optional[TransFn] = None,
                   allow_fail: bool = False):
            param_key = next(k for k, v in to_params.items() if v is param)
            # Delete regardless of whether weight exists.
            del to_params[param_key]
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
                        layer: RelativeMultiheadAttention
                        layer = xlnet.attn_layers[idx]
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
                assign(layer.output_projection.weight,  # DO NOT TRANSPOSE!!!!
                       prefix + "o/kernel", lambda p: p.view(p.size(0), -1))
                assign_layer_norm(layer.layer_norm, prefix)

            for idx in range(n_layers):
                layer: PositionWiseFF = xlnet.ff_layers[idx]
                prefix = f"transformer/layer_{idx}/ff/"
                for linear_idx in range(1, 2 + 1):
                    linear_prefix = f"{prefix}layer_{linear_idx}/"
                    linear_layer: nn.Linear = getattr(
                        layer, f"linear{linear_idx}")
                    assign_linear(linear_layer, linear_prefix)
                assign_layer_norm(layer.layer_norm, prefix)

            seg_embeds = [
                p.squeeze(0)
                for p in torch.chunk(
                    get_weight("transformer/seg_embed"), n_layers, dim=0)]
            for idx in range(n_layers):
                assign(xlnet.attn_layers[idx].segment_embed, seg_embeds[idx])

            if hasattr(xlnet, 'mask_emb') and hasattr(xlnet, 'lm_bias'):
                assign(xlnet.mask_emb, "transformer/mask_emb/mask_emb")
                assign(xlnet.lm_bias, "lm_loss/bias")

        load_xlnet_model(self)

        if len(from_params) > 0:
            print(f"WARNING: Certain weights from checkpoint are not loaded: "
                  f"{list(from_params.keys())}")

        filtered_to_params = [k for k in to_params if k.startswith("xlnet")]
        if len(filtered_to_params) > 0:
            print(f"WARNING: Certain parameters are not initialized: "
                  f"{list(filtered_to_params)}")
