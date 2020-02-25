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
Custom initializers used in various methods.
"""

import math

import torch


def variance_scaling_initializer(inputs: torch.Tensor,
                                 factor: float = 2.0, mode: str = 'FAN_IN',
                                 uniform: bool = False):
    r"""Returns an initializer that generates tensors without scaling variance.
    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. This initializer use the following formula:
    ```python
        if mode='FAN_IN': # Count only number of input connections.
        n = fan_in
        elif mode='FAN_OUT': # Count only number of output connections.
        n = fan_out
        elif mode='FAN_AVG': # Average number of inputs and output connections.
        n = (fan_in + fan_out)/2.0
        truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
    ```
    * To get [Delving Deep into Rectifiers](
        http://arxiv.org/pdf/1502.01852v1.pdf) (also know as the "MSRA
        initialization"), use (Default):<br/>
        `factor=2.0 mode='FAN_IN' uniform=False`
    * To get [Convolutional Architecture for Fast Feature Embedding](
        http://arxiv.org/abs/1408.5093), use:<br/>
        `factor=1.0 mode='FAN_IN' uniform=True`
    * To get [Understanding the difficulty of training deep feed-forward
        neural networks](
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
        use:<br/>
        `factor=1.0 mode='FAN_AVG' uniform=True.`
    * To get `xavier_initializer` use either:<br/>
        `factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
        `factor=1.0 mode='FAN_AVG' uniform=False`.
    Args:
        factor: Float.  A multiplicative factor.
        mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
        uniform: Whether to use uniform or normal distributed
                 random initialization.
    Returns:
        An initializer that generates tensors with unit variance.
    Raises:
        ValueError: if `dtype` is not a floating point type.
        TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
    """

    # Estimating fan_in and fan_out is not possible to do perfectly, but we
    # try. This is the right thing for matrix multiply and convolutions.
    shape = inputs.size()
    fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
    fan_out = float(shape[-1])
    for dim in shape[:-2]:
        fan_in *= float(dim)
        fan_out *= float(dim)

    if mode == 'FAN_IN':
        # Count only number of input connections.
        n = fan_in
    elif mode == 'FAN_OUT':
        # Count only number of output connections.
        n = fan_out
    elif mode == 'FAN_AVG':
        # Average number of inputs and output connections.
        n = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(f"Unknown mode {mode} [FAN_IN, FAN_OUT, FAN_AVG]")

    if uniform:
        # To get stddev = math.sqrt(factor / n) need to
        # adjust for uniform.
        limit = math.sqrt(3.0 * factor / n)
        inputs.data.uniform_(-limit, limit)
    else:
        # To get stddev = math.sqrt(factor / n) need to
        # adjust for truncated normal.
        trunc_stddev = math.sqrt(1.3 * factor / n)

        u1 = torch.rand(shape) * (1 - math.exp(-2)) + math.exp(-2)
        u2 = torch.rand(shape)
        rnd = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * math.pi * u2)

        ret = rnd * trunc_stddev
        inputs.data = ret
