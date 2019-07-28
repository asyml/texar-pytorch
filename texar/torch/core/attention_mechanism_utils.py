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
"""Various helper utilities for attention mechanism.

The implementation of `sparsemax` adapted from:
    `https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/
    modules/sparse_activations.py`
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.autograd import Function

from texar.torch.utils.utils import sequence_mask

__all__ = [
    'hardmax',
    'maybe_mask_score',
    'prepare_memory',
    'safe_cumprod',
    'sparsemax',
]


def hardmax(logits: torch.Tensor) -> torch.Tensor:
    r"""Returns batched one-hot vectors. The depth index containing
    the `1` is that of the maximum logit value.

    Args:
        logits: A batch tensor of logit values.

    Returns:
        A batched one-hot tensor.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    depth = logits.shape[-1]
    one_hot = torch.eye(depth, dtype=torch.int64)
    return F.embedding(torch.argmax(logits, -1), one_hot)


def maybe_mask_score(score: torch.Tensor,
                     score_mask_value: torch.Tensor,
                     memory_sequence_length: Optional[torch.LongTensor]) \
        -> torch.Tensor:
    r"""Mask the attention score based on the masks."""
    if memory_sequence_length is None:
        return score

    for memory_sequence_length_value in memory_sequence_length:
        if memory_sequence_length_value <= 0:
            raise ValueError(
                "All values in memory_sequence_length must be greater "
                "than zero.")

    score_mask = sequence_mask(memory_sequence_length,
                               max_len=score.shape[1])
    score_mask_values = score_mask_value * torch.ones_like(score)
    return torch.where(score_mask, score, score_mask_values)


def prepare_memory(memory: torch.Tensor,
                   memory_sequence_length: Optional[torch.LongTensor]) \
        -> torch.Tensor:
    r"""Convert to tensor and possibly mask ``memory``.

    Args:
        memory: tensor, shaped ``[batch_size, max_time, ...]``.
        memory_sequence_length: integer tensor, shaped ``[batch_size]``.

    Returns:
        A (possibly masked), new ``memory``.

    Raises:
        ValueError: if ``memory`` and ``memory_sequence_length`` do not have
        the same ``batch_size``.
    """
    if (memory_sequence_length is not None and
            not isinstance(memory_sequence_length, torch.Tensor)):
        memory_sequence_length = torch.tensor(memory_sequence_length,
                                              dtype=torch.long,
                                              device=memory.device)

    if memory_sequence_length is None:
        seq_len_mask = None
    else:
        seq_len_mask = sequence_mask(memory_sequence_length,
                                     max_len=memory.shape[1],
                                     dtype=memory.dtype)
        seq_len_batch_size = memory_sequence_length.shape[0]

    # Mask the memory based on the memory mask.
    rank = memory.dim()
    m_batch_size = memory.shape[0]

    if seq_len_mask is not None:
        if seq_len_batch_size != m_batch_size:
            raise ValueError("memory_sequence_length and memory tensor "
                             "batch sizes do not match.")
        return memory * seq_len_mask.view(
            seq_len_mask.size() + (1,) * (rank - 2))
    else:
        return memory


def safe_cumprod(x: torch.Tensor,
                 *args,
                 **kwargs) -> torch.Tensor:
    r"""Computes cumprod of x in logspace using cumsum to avoid underflow.
    The cumprod function and its gradient can result in numerical
    instabilities when its argument has very small and/or zero values.
    As long as the argument is all positive, we can instead compute the
    cumulative product as `exp(cumsum(log(x)))`.  This function can be called
    identically to :torch:`cumprod`.

    Args:
        x: Tensor to take the cumulative product of.
        *args: Passed on to cumsum; these are identical to those in cumprod.
        **kwargs: Passed on to cumsum; these are identical to those in cumprod.

    Returns:
        Cumulative product of x.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    tiny = torch.finfo(x.dtype).tiny

    return torch.exp(torch.cumsum(torch.log(torch.clamp(x, tiny, 1)),
                                  *args, **kwargs))


def _make_ix_like(input: torch.Tensor,
                  dim: int = -1) -> torch.Tensor:
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = (-1,) + (1,) * (input.dim() - 1)
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input: torch.Tensor,
                           dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""`Sparsemax` building block: compute the threshold.

    Args:
        input: any dimension
        dim: dimension along which to apply `sparsemax`.

    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx,  # type: ignore
                input: torch.Tensor,
                dim: int = -1) -> torch.Tensor:
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx,  # type: ignore
                 grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""`sparsemax`: normalizing sparse transform (a la softmax).

    Args:
        input (Tensor): A batch tensor of logit values.
        dim: Dimension along which to apply `sparsemax`.

    Returns:
        Tensor: output with the same shape as input.
    """
    return SparsemaxFunction.apply(input, dim)  # type: ignore
