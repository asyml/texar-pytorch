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
Utilities for loading TensorFlow checkpoints.
"""

from typing import (
    Callable, Dict, Optional, Union)

import numpy as np
import torch
from torch import nn

from xlnet.model.model import XLNet, XLNetSummary
from xlnet.model.modules import PositionWiseFF, RelativeMultiheadAttention

__all__ = [
    "load_from_tf_checkpoint",
]


def load_from_tf_checkpoint(model: Union[XLNet, XLNetSummary], path: str,
                            task_name: Optional[str] = None):
    r"""Load pre-trained model weights from a TensorFlow checkpoint.

    Args:
        model: The `XLNet` model to load weights into. Can be an instance of
            :class:`XLNet`, :class:`XLNetClassifier`, or
            :class:`XLNetRegressor`.
        path (str): Path to the TF checkpoint. Note that a TF checkpoint is a
            collection of 3 or more files, including ``.meta``, ``.index``, and
            ``.data-00000-of-...`` files.
        task_name (str, optional): If specified, load task specific parameters
            from the model. The model must be of the appropriate class
            (:class:`XLNetClassifier` or :class:`XLNetRegressor`). The task name
            should be identical to the ``task_name`` flag used in TF training.
    """
    # remember to call .contiguous after trans_fn
    import tensorflow as tf
    ckpt = tf.train.load_checkpoint(path)
    from_params: Dict[str, np.ndarray] = {
        key: ckpt.get_tensor(key)
        for key in ckpt.get_variable_to_shape_map().keys()}
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

    def load_xlnet_model(xlnet):
        for bias_name in ['r_r_bias', 'r_w_bias', 'r_s_bias']:
            assign(getattr(xlnet, bias_name), "transformer/" + bias_name)
        assign(xlnet.word_embed.weight,
               "transformer/word_embedding/lookup_table")

        n_layers = len(xlnet.attn_layers)
        for idx in range(n_layers):
            layer: RelativeMultiheadAttention = xlnet.attn_layers[idx]
            prefix = f"transformer/layer_{idx}/rel_attn/"
            qkv_weights = [get_weight(prefix + f"{part}/kernel")
                           for part in "qkv"]
            assign(layer.head_projection.weight,
                   torch.cat([
                       p.view(p.size(0), -1) for p in qkv_weights
                   ], dim=1).t())
            trans_fn = lambda p: p.view(p.size(0), -1).t()
            assign(layer.pos_projection.weight, prefix + "r/kernel", trans_fn)
            assign(layer.output_projection.weight,  # DO NOT TRANSPOSE THIS!!!!
                   prefix + "o/kernel", lambda p: p.view(p.size(0), -1))
            assign(layer.layer_norm.weight, prefix + "LayerNorm/gamma")
            assign(layer.layer_norm.bias, prefix + "LayerNorm/beta")

        for idx in range(n_layers):
            layer: PositionWiseFF = xlnet.ff_layers[idx]
            prefix = f"transformer/layer_{idx}/ff/"
            for linear_idx in range(1, 2 + 1):
                linear_prefix = f"{prefix}layer_{linear_idx}/"
                linear_layer: nn.Linear = getattr(layer, f"linear{linear_idx}")
                trans_fn = lambda p: p.t()
                assign(linear_layer.weight, linear_prefix + "kernel", trans_fn)
                assign(linear_layer.bias, linear_prefix + "bias")
            assign(layer.layer_norm.weight, prefix + "LayerNorm/gamma")
            assign(layer.layer_norm.bias, prefix + "LayerNorm/beta")

        seg_embeds = [
            p.squeeze(0)
            for p in torch.chunk(
                get_weight("transformer/seg_embed"), n_layers, dim=0)]
        for idx in range(n_layers):
            assign(xlnet.attn_layers[idx].segment_embed, seg_embeds[idx])

    if isinstance(model, XLNetSummary):
        trans_fn = lambda p: p.t()
        if task_name is not None:
            prefix = "sequnece_summary/summary/"
            assign(model.projection.weight, prefix + "kernel", trans_fn)
            assign(model.projection.bias, prefix + "bias")
            prefix = f"classification_{task_name}/logit/"
            assign(model.hidden_to_logits.weight, prefix + "kernel", trans_fn)
            assign(model.hidden_to_logits.bias, prefix + "bias")
        load_xlnet_model(model.xlnet)
    elif isinstance(model, XLNet):
        if task_name is not None:
            raise ValueError("model must be an instance of XLNetClassifier or "
                             "XLNetRegressor when task_name is not None")
        load_xlnet_model(model)
    else:
        raise ValueError("The specified model must be an XLNet model.")

    if len(from_params) > 0:
        print(f"WARNING: Certain weights from checkpoint are not loaded: "
              f"{list(from_params.keys())}")
    if task_name is not None and len(to_params) > 0:
        print(f"WARNING: Certain parameters are not initialized: "
              f"{list(to_params.keys())}")
    else:
        filtered_to_params = [k for k in to_params if k.startswith("xlnet")]
        if len(filtered_to_params) > 0:
            print(f"WARNING: Certain parameters are not initialized: "
                  f"{list(filtered_to_params)}")
