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
Executor metrics for regression tasks.
"""

import math
from typing import Sequence

from texar.torch.run.metric.base_metric import StreamingMetric

__all__ = [
    "PearsonR",
    "RMSE",
]


class PearsonR(StreamingMetric[float, float]):
    r"""The Pearson correlation coefficient (Pearson's r) metric for evaluation
    regression tasks. Pearson's r is a measure of linear correlation between two
    sets of variables. Pearson's r ranges between -1 and 1, with 1 indicating
    total positive linear correlation, -1 indicating total negative linear
    correlation, and 0 indication no linear correlation.

    Pearson's r is a :class:`~texar.torch.run.metric.StreamingMetric`, requires
    both predicted values and labels. Pearson's r values are :class:`float`
    numbers between -1 and 1, with higher values being better.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
    x_sum: float
    x2_sum: float
    y_sum: float
    y2_sum: float
    xy_sum: float

    def reset(self) -> None:
        super().reset()
        self.x_sum = self.y_sum = 0.0
        self.x2_sum = self.y2_sum = 0.0
        self.xy_sum = 0.0

    def add(self, xs: Sequence[float], ys: Sequence[float]):
        super().add(xs, ys)
        self.x_sum += sum(xs)
        self.x2_sum += sum(x * x for x in xs)
        self.y_sum += sum(ys)
        self.y2_sum += sum(y * y for y in ys)
        self.xy_sum += sum(x * y for x, y in zip(xs, ys))

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        numerator = self.xy_sum - self.x_sum * self.y_sum / self.count
        denominator_x = self.x2_sum - self.x_sum ** 2 / self.count
        denominator_y = self.y2_sum - self.y_sum ** 2 / self.count
        if denominator_x == 0.0 or denominator_y == 0.0:
            return math.nan
        return numerator / math.sqrt(denominator_x * denominator_y)


class RMSE(StreamingMetric[float, float]):
    r"""The root mean squared error (RMSE) metric for evaluation regression
    tasks. RMSE is defined as the standard deviation of the residuals
    (difference between predicted values and ground truth values).

    RMSE is a :class:`~texar.torch.run.metric.StreamingMetric`, requires both
    predicted values and labels. RMSE values are :class:`float` numbers with a
    lower bound of 0. Lower values are better.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
    higher_is_better = False

    squared_sum: float

    def reset(self) -> None:
        super().reset()
        self.squared_sum = 0.0

    def add(self, predicted: Sequence[float], labels: Sequence[float]) -> None:
        super().add(predicted, labels)
        self.squared_sum += sum((x - y) ** 2 for x, y in zip(predicted, labels))

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return math.sqrt(self.squared_sum / self.count)
