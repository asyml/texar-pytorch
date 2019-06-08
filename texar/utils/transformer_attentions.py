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
Attentions specific to Transformer.
"""

from typing import Optional, Tuple

import numpy as np
import torch

__all__ = [
    'attention_bias_lower_triangle',
    'attention_bias_ignore_padding',
    'attention_bias_local',
]


def attention_bias_lower_triangle(length: int,
                                  bias_value: float = -1e18) -> torch.Tensor:
    r"""Create an bias tensor to be added to attention logits.
    Allows a query to attend to all positions up to and including its own.

    Args:
        length: a scalar.
        bias_value: value to fill the bias tensor with.

    Returns:
        a ``Tensor`` with shape [1, 1, length, length].
    """
    return attention_bias_local(length, -1, 0, bias_value)


def attention_bias_local(length: int, max_backward: int, max_forward: int,
                         bias_value: float = -1e18) -> torch.Tensor:
    r"""Create an bias tensor to be added to attention logits.
    A position may attend to positions at most max_distance from it,
    forward and backwards.

    This does not actually save any computation.

    Args:
        length: int
        max_backward: int, maximum distance backward to attend. Negative
            values indicate unlimited.
        max_forward: int, maximum distance forward to attend. Negative
            values indicate unlimited.
        bias_value: value to fill the bias tensor with.

    Returns:
        a ``Tensor`` with shape [1, 1, length, length].
        [batch_size, num_heads, query_len, query_len]
    """
    band = _ones_matrix_band_part(
        length,
        length,
        max_backward,
        max_forward,
        out_shape=(1, 1, length, length))
    return bias_value * (1.0 - band)


def attention_bias_ignore_padding(memory_padding: torch.Tensor,
                                  bias_value: float = -1e18) -> torch.Tensor:
    r"""Create an bias tensor to be added to attention logits.

    Args:
        memory_padding: a float ``Tensor`` with shape [batch, memory_length].
        bias_value: value to fill the bias tensor with.

    Returns:
        a ``Tensor`` with shape [batch, 1, 1, memory_length].
        each dim corresponding to batch_size, num_heads, queries_len,
        memory_length
    """
    ret = memory_padding * bias_value
    return ret.view(ret.size(0), 1, 1, ret.size(-1))


def _ones_matrix_band_part(rows: int, cols: int, num_lower: int, num_upper: int,
                           out_shape: Optional[Tuple[int, ...]] = None) \
        -> torch.Tensor:
    r"""Matrix band part of ones.
    """
    if num_lower < 0:
        num_lower = rows - 1
    if num_upper < 0:
        num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
        band = band.reshape(out_shape)
    band = torch.as_tensor(band, dtype=torch.float32)
    return band
