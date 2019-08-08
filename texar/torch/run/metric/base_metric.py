from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Sequence, TypeVar

__all__ = [
    "Metric",
    "SimpleMetric",
    "StreamingMetric",
]

Input = TypeVar('Input')
Value = TypeVar('Value')


class Metric(Generic[Input, Value], ABC):
    higher_is_better: bool = True
    requires_pred: bool = True
    requires_label: bool = True

    def __init__(self, *, pred_name: Optional[str],
                 label_name: Optional[str] = "label",
                 higher_is_better: Optional[bool] = None):
        self.reset()
        if self.requires_label and label_name is None:
            raise ValueError(f"Metric {self.metric_name} requires a "
                             f"label name, but None is provided")
        if higher_is_better is not None:
            self.higher_is_better = higher_is_better
        if not self.requires_pred:
            pred_name = None
        if not self.requires_label:
            label_name = None
        self._pred_name = pred_name
        self._label_name = label_name

    @property
    def metric_name(self) -> str:
        return self.__class__.__name__

    @property
    def pred_name(self) -> Optional[str]:
        return self._pred_name

    @property
    def label_name(self) -> Optional[str]:
        return self._label_name

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
        raise NotImplementedError

    @abstractmethod
    def value(self) -> Value:
        raise NotImplementedError

    def better(self, cur: Value, prev: Value) -> Optional[bool]:
        result = (True if cur > prev else  # type: ignore
                  False if cur < prev else None)  # type: ignore
        if not self.higher_is_better and result is not None:
            result = not result
        return result


class SimpleMetric(Metric[Input, Value], ABC):
    labels: List[Input]
    predicted: List[Input]

    def reset(self) -> None:
        self.labels = []
        self.predicted = []

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]):
        self.predicted.extend(predicted)
        self.labels.extend(labels)


class StreamingMetric(Metric[Input, Value], ABC):
    count: int

    def reset(self) -> None:
        self.count = 0

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
        if len(predicted) != len(labels):
            raise ValueError(
                "Lists 'predicted' and 'labels' should have the same length")
        self.count += len(predicted)
