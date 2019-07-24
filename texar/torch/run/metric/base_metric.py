from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

__all__ = [
    "Metric",
    "SimpleMetric",
    "StreamingMetric",
]

Input = TypeVar('Input')
Value = TypeVar('Value')


class Metric(Generic[Input, Value], ABC):
    higher_is_better: bool = True

    def __init__(self, *, pred_name: str = "loss", label_name: str = "label"):
        self.reset()
        self._pred_name = pred_name
        self._label_name = label_name

    @property
    def pred_name(self) -> str:
        return self._pred_name

    @property
    def label_name(self) -> str:
        return self._label_name

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add(self, predicted: List[Input], labels: List[Input]) -> None:
        raise NotImplementedError

    @abstractmethod
    def value(self) -> Value:
        raise NotImplementedError

    def better(self, a: Value, b: Value) -> bool:
        if self.higher_is_better:
            return a > b
        return a < b


class SimpleMetric(Metric[Input, Value], ABC):
    labels: List[Input]
    predicted: List[Input]

    def reset(self) -> None:
        self.labels = []
        self.predicted = []

    def add(self, predicted: List[Input], labels: List[Input]):
        self.predicted.extend(predicted)
        self.labels.extend(labels)


class StreamingMetric(Metric[Input, Value], ABC):
    count: int

    def reset(self) -> None:
        self.count = 0

    def add(self, predicted: List[Input], labels: List[Input]) -> None:
        if len(predicted) != len(labels):
            raise ValueError(
                "Lists 'predicted' and 'labels' should have the same length")
        self.count += len(predicted)
