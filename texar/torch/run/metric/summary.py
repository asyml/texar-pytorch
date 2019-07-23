from collections import deque
from typing import List, Deque

import numpy as np

from texar.torch.run.metric.base_metric import StreamingMetric


class Average(StreamingMetric[float, float]):
    sum: float

    def reset(self) -> None:
        self.count = 0
        self.sum = 0.0

    def add(self, predicted: List[float], _) -> None:
        self.count += len(predicted)
        self.sum += sum(predicted)

    def value(self) -> float:
        return self.sum / self.count


class AveragePerplexity(Average):
    def add(self, predicted: List[float], _) -> None:
        super().add(np.exp(predicted), _)


class RunningAverage(StreamingMetric[float, float]):
    history: Deque[float]
    sum: float

    def __init__(self, queue_size: int, **kwargs):
        super().__init__(**kwargs)
        self.queue_size = queue_size

    def reset(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.history = deque()

    def add(self, predicted: List[float], _) -> None:
        if len(predicted) >= self.queue_size:
            self.history = deque(predicted[:-self.queue_size])
            self.sum = sum(self.history)
        else:
            for val in predicted:
                self.sum += val - self.history.popleft()
                self.history.append(val)

    def value(self) -> float:
        return self.sum / len(self.history)
