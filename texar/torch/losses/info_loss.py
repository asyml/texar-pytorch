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
Infomational Theory Losses.
"""
import torch
import torch.nn.functional as F


__all__ = [
    "kl_divg_loss_with_logits",
]


def kl_divg_loss_with_logits(
    tgt_log_probs: torch.Tensor,
    input_log_probs: torch.Tensor,
    softmax_temp: float = 1,
    confidence_thres: float = -1,
    reduction: str = "mean",
):
    r"""
    This function calculates the Kullback-Leibler divergence
    between a pair of distributions.

    It supports confidence-based masking and distribution sharpening.
    Please refer to the UDA paper for more details.
    (https://arxiv.org/abs/1904.12848)
    Args:
        - tgt_log_probs:
            The target logits.
        - input_log_probs:
            The input logits.
        - softmax_temp: The softmax temparature for sharpening the distribution.
        - confidence_thres: The threshold for confidence-masking.
        - reduction: The reduction method for torch loss, refer to Pytorch doc of
            torch.nn.functional.kl_div for details.
    Returns:
        A loss term in Torch.
    """
    # Sharpening the target distribution.
    with torch.no_grad():
        tgt_probs: torch.Tensor = F.softmax(
            tgt_log_probs / softmax_temp,
            axis=-1
        )

    # Calculate the KL divergence.
    loss = F.kl_div(input_log_probs, tgt_probs, reduction=reduction)

    if confidence_thres != -1:
        # Mask the training sample based on confidence.
        largest_prob, _ = torch.max(
            F.softmax(input_log_probs, axis=-1),
            dim=-1
        )
        loss_mask: torch.Tensor = torch.gt(
            largest_prob,
            confidence_thres
        ).double()
        loss *= loss_mask

    return loss
