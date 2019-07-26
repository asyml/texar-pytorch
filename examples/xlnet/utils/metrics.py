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
Evaluation metrics.
"""

import math
from typing import TypeVar, Generic, List

__all__ = [
    "StreamingMetric",
    "StreamingAccuracy",
    "StreamingPearsonR",
    "accuracy",
    "pearsonr",
]

T = TypeVar('T')


class StreamingMetric(Generic[T]):
    name: str

    def __init__(self):
        self.count = 0

    def add(self, gold: List[T], pred: List[T]):
        assert len(gold) == len(pred)
        self.count += len(gold)

    def value(self) -> float:
        raise NotImplementedError


class StreamingAccuracy(StreamingMetric[T]):
    name = "accuracy"

    def __init__(self):
        super().__init__()
        self.correct = 0

    def add(self, gold: List[T], pred: List[T]):
        super().add(gold, pred)
        self.correct += sum(int(a == b) for a, b in zip(gold, pred))

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.correct / self.count


class StreamingPearsonR(StreamingMetric[float]):
    name = "pearsonR"

    def __init__(self):
        super().__init__()
        self.x_sum = 0.0
        self.x2_sum = 0.0
        self.y_sum = 0.0
        self.y2_sum = 0.0
        self.xy_sum = 0.0

    def add(self, xs: List[float], ys: List[float]):
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


def accuracy(gold: List[int], pred: List[int]) -> float:
    return sum(int(a == b) for a, b in zip(gold, pred)) / len(gold)


def pearsonr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    x_avg = sum(xs) / n
    y_avg = sum(ys) / n
    numerator = sum((x - x_avg) * (y - y_avg) for x, y in zip(xs, ys))
    denominator_x = sum((x - x_avg) ** 2 for x in xs)
    denominator_y = sum((y - y_avg) ** 2 for y in ys)
    if denominator_x == 0.0 or denominator_y == 0.0:
        return math.nan
    return numerator / math.sqrt(denominator_x * denominator_y)
