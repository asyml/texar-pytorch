# Copyright 2020 The Texar Authors. All Rights Reserved.
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
Informational Theory Losses.
"""
import torch
import torch.nn.functional as F


__all__ = [
    "kl_divg_loss_with_logits",
]


def kl_divg_loss_with_logits(
    target_logits: torch.Tensor,
    input_logits: torch.Tensor,
    softmax_temperature: float = 1,
    confidence_threshold: float = -1,
    reduction: str = "mean",
):
    r"""
    This function calculates the Kullback-Leibler divergence
    between a pair of logits of the distributions.

    It supports confidence-based masking and distribution sharpening.
    Please refer to the UDA paper for more details.
    (https://arxiv.org/abs/1904.12848)
    Args:
        - target_logits (Tensor):
            A Tensor of arbitrary shape containing the target logits.
        - input_logits (Tensor):
            A Tensor of the same shape as 'target_logits'
            containing the input logits.
        - softmax_temperature (float, optional):
            The softmax temperature for sharpening the distribution.
        - confidence_threshold (float, optional):
            The threshold for confidence-masking. It is a threshold
            of the probability in [0, 1], rather than of the logit.
            If set to -1, the threshold will be ignored.
        - reduction (optional):
            Default: 'mean'.
            This is the same as the `reduction` argument in :torch_docs:
            `torch.nn.functional.kl_div <https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.kl_div>`  # pylint: disable=line-too-long
            Specifies the reduction to apply to the output:

            - :attr:`'none'`: no reduction will be applied.
            - :attr:`'batchmean'`: the sum of the output will be
                divided by the batchsize.
            - :attr:`'sum'`: the output will be summed.
            - :attr:`'mean'`: the output will be divided by
                the number of elements in the output.
    Returns:
        The loss, as a pytorch scalar float tensor.
    """  # noqa
    # Sharpening the target distribution.
    with torch.no_grad():
        target_probs: torch.Tensor = F.softmax(
            target_logits / softmax_temperature,
            dim=-1
        )

    input_log_probs = F.log_softmax(
        input_logits,
        dim=-1
    )
    # Calculate the KL divergence.
    # No reduction before the confidence masking.
    loss = F.kl_div(input_log_probs, target_probs, reduction="none")

    if confidence_threshold != -1:
        # Mask the training sample based on confidence.
        largest_prob, _ = torch.max(
            F.softmax(input_logits, dim=-1),
            dim=-1
        )
        loss_mask: torch.Tensor = torch.gt(
            largest_prob,
            confidence_threshold
        ).float().unsqueeze(-1)
        loss *= loss_mask

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "batchmean":
        return loss.sum() / loss.size(0)
    if reduction != "none":
        raise ValueError(
            f"The reduction method {reduction} is not supported!"
        )

    return loss
