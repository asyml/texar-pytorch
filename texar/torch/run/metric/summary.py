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
Executor metrics for summaries.
"""

from collections import deque
from typing import Any, Deque, Optional, Sequence

import numpy as np
from torch.optim.optimizer import Optimizer

from texar.torch.run.metric.base_metric import StreamingMetric

__all__ = [
    "Average",
    "AveragePerplexity",
    "RunningAverage",
    "LR",
]


class Average(StreamingMetric[float, float]):
    r"""The average of a specific predicted value.

    Average is a :class:`~texar.torch.run.metric.StreamingMetric`, requires only
    predicted values. Average values are unbounded :class:`float` numbers. By
    default, lower values are better, but the behavior can be configured.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model. Defaults to ``"loss"``.
        higher_is_better (bool, optional): If specified, the
            :attr:`higher_is_better` attribute for the instance is overwritten
            by the specified value. Defaults to `False`.
    """
    higher_is_better = False
    requires_label = False

    sum: float

    def __init__(self, *, pred_name: str = "loss",
                 higher_is_better: bool = False):
        super().__init__(pred_name=pred_name, higher_is_better=higher_is_better)

    def reset(self) -> None:
        super().reset()
        self.sum = 0.0

    def add(self, predicted: Sequence[float], _) -> None:
        self.count += len(predicted)
        self.sum += sum(predicted)

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class AveragePerplexity(Average):
    # TODO: Create a `WeightedAverage` class that takes `(value, weight)`
    #   and subclass that instead.
    higher_is_better = False

    def add(self, predicted: Sequence[float], _) -> None:
        super().add(np.exp(predicted), _)


class RunningAverage(StreamingMetric[float, float]):
    r"""The running average of a specific predicted value, i.e., the average
    computed over the most recent :attr:`queue_size` values.

    Running average is a :class:`~texar.torch.run.metric.StreamingMetric`,
    requires only predicted values. Running average values are unbounded
    :class:`float` numbers. By default, lower values are better, but the
    behavior can be configured.

    Keyword Args:
        queue_size (int): Size of the queue to keep history values. The running
            average is computed over the most recent :attr:`queue_size` values.
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model. Defaults to ``"loss"``.
        higher_is_better (bool, optional): If specified, the
            :attr:`higher_is_better` attribute for the instance is overwritten
            by the specified value.
    """
    higher_is_better = False
    requires_label = False

    history: Deque[float]
    sum: float

    def __init__(self, queue_size: int, *, pred_name: str = "loss",
                 higher_is_better: bool = False):
        super().__init__(pred_name=pred_name, higher_is_better=higher_is_better)
        if not isinstance(queue_size, int) or queue_size <= 0:
            raise ValueError("'queue_size' must be a position integer")
        self.queue_size = queue_size

    def reset(self) -> None:
        super().reset()
        self.sum = 0.0
        self.history = deque()

    def add(self, predicted: Sequence[float], _) -> None:
        if len(predicted) >= self.queue_size:
            self.history = deque(predicted[-self.queue_size:])
            self.sum = sum(self.history)
        else:
            for _ in range(len(predicted) -
                           (self.queue_size - len(self.history))):
                self.sum -= self.history.popleft()
            self.sum += sum(predicted)
            self.history.extend(predicted)

    def value(self) -> float:
        if len(self.history) == 0:
            return 0.0
        return self.sum / len(self.history)


class LR(StreamingMetric[Any, float]):
    r"""The learning rate (LR) of the given optimizer. This is not exactly a
    metric, but rather a convenience object to print learning rates in log.

    LR is a :class:`~texar.torch.run.metric.StreamingMetric`, requires neither
    predicted values nor labels. LR values are unbounded :class:`float` numbers,
    with no clear definition of "better". Comparison of two learning rates are
    not meaningful.

    Keyword Args:
        optimizer: The optimizer instance.
        param_group (int, optional): Index of the parameter group to obtain the
            learning rate from. Defaults to 0. You don't need to specify this if
            the optimizer contains only one parameter group (e.g., constructed
            using :python:`optim_class(model.parameters())`.
    """

    requires_pred = False
    requires_label = False

    def __init__(self, optimizer: Optimizer, param_group: int = 0):
        super().__init__(pred_name=None)
        self.optimizer = optimizer
        self.group = param_group

    def add(self, _, __):
        pass

    def value(self) -> float:
        return self.optimizer.param_groups[self.group]['lr']  # type: ignore

    def better(self, cur: float, prev: float) -> Optional[bool]:
        # Always return `None` to indicate values are uncomparable.
        return None
