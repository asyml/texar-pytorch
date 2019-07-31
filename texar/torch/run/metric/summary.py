from collections import deque
from typing import List, Deque

import numpy as np

from texar.torch.run.metric.base_metric import StreamingMetric


class Average(StreamingMetric[float, float]):
    requires_label = False

    sum: float

    def __init__(self, *, pred_name: str = "loss", **kwargs):
        if pred_name == "loss" and kwargs.get("higher_is_better", None) is None:
            kwargs["higher_is_better"] = False
        super().__init__(pred_name=pred_name, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.sum = 0.0

    def add(self, predicted: List[float], _) -> None:
        self.count += len(predicted)
        self.sum += sum(predicted)

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class AveragePerplexity(Average):
    higher_is_better = False

    def add(self, predicted: List[float], _) -> None:
        super().add(np.exp(predicted), _)


class RunningAverage(StreamingMetric[float, float]):
    history: Deque[float]
    sum: float

    def __init__(self, queue_size: int, *, pred_name: str = "loss", **kwargs):
        if pred_name == "loss" and kwargs.get("higher_is_better", None) is None:
            kwargs["higher_is_better"] = False
        super().__init__(pred_name=pred_name, **kwargs)
        if not isinstance(queue_size, int) or queue_size <= 0:
            raise ValueError("'queue_size' must be a position integer")
        self.queue_size = queue_size

    def reset(self) -> None:
        super().reset()
        self.sum = 0.0
        self.history = deque()

    def add(self, predicted: List[float], _) -> None:
        if len(predicted) >= self.queue_size:
            self.history = deque(predicted[:-self.queue_size])
            self.sum = sum(self.history)
        else:
            for val in predicted:
                self.sum += val
                if len(self.history) == self.queue_size:
                    self.sum -= self.history.popleft()
                self.history.append(val)

    def value(self) -> float:
        if len(self.history) == 0:
            return 0.0
        return self.sum / len(self.history)
