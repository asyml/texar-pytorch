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
Various metrics.
"""

from typing import Optional

import torch

__all__ = [
    "accuracy",
    "binary_clas_accuracy",
]


def accuracy(labels: torch.Tensor,
             preds: torch.Tensor) -> torch.Tensor:
    r"""Calculates the accuracy of predictions.

    Args:
        labels: The ground truth values. A Tensor of the same shape of
            :attr:`preds`.
        preds: A Tensor of any shape containing the predicted values.

    Returns:
        A float scalar Tensor containing the accuracy.
    """
    labels = labels.type(preds.dtype).reshape(preds.shape)
    return (labels == preds).float().mean()


def binary_clas_accuracy(pos_preds: Optional[torch.Tensor] = None,
                         neg_preds: Optional[torch.Tensor] = None) -> \
        Optional[torch.Tensor]:
    r"""Calculates the accuracy of binary predictions.

    Args:
        pos_preds (optional): A Tensor of any shape containing the
            predicted values on positive data (i.e., ground truth labels are 1).
        neg_preds (optional): A Tensor of any shape containing the
            predicted values on negative data (i.e., ground truth labels are 0).

    Returns:
        A float scalar Tensor containing the accuracy.
    """
    if pos_preds is None and neg_preds is None:
        return None
    if pos_preds is not None:
        pos_accu = accuracy(torch.ones_like(pos_preds), pos_preds)
        psize = float(torch.numel(pos_preds))
    else:
        pos_accu = torch.tensor(0.0)
        psize = torch.tensor(0.0)
    if neg_preds is not None:
        neg_accu = accuracy(torch.zeros_like(neg_preds), neg_preds)
        nsize = float(torch.numel(neg_preds))
    else:
        neg_accu = torch.tensor(0.0)
        nsize = torch.tensor(0.0)
    accu = (pos_accu * psize + neg_accu * nsize) / (psize + nsize)
    return accu
