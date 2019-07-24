from abc import ABC
from typing import Dict, List, Optional, TypeVar

import numpy as np

from texar.torch.run.metric.base_metric import StreamingMetric

__all__ = [
    "Average",
    "ConfusionMatrix",
    "Precision",
    "Recall",
    "F1",
]

Input = TypeVar('Input')
Value = TypeVar('Value')


class Average(StreamingMetric[Input, float]):
    correct: int

    def reset(self) -> None:
        super().reset()
        self.correct = 0

    def add(self, predicted: List[Input], labels: List[Input]) -> None:
        super().add(predicted, labels)
        self.count += len(predicted)
        self.correct += sum(int(a == b) for a, b in zip(predicted, labels))

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.correct / self.count


class _ConfusionMatrix(StreamingMetric[Input, Value], ABC):
    count: int
    matrix: Optional[np.ndarray]  # matrix[pred][label]
    pred_count: List[int]
    label_count: List[int]
    class_id: Dict[Input, int]

    def reset(self) -> None:
        super().reset()
        self.matrix = None
        self.pred_count = []
        self.label_count = []
        self.class_id = {}

    def _convert_ids(self, classes: List[Input]) -> List[int]:
        ids = []
        cnt = 0
        for klass in classes:
            if klass not in self.class_id:
                self.class_id[klass] = len(self.class_id)
                cnt += 1
            ids.append(self.class_id[klass])
        if self.matrix is None:
            self.matrix = np.zeros((cnt, cnt), dtype=np.int)
        else:
            self.matrix = np.pad(self.matrix, [(0, cnt), (0, cnt)],
                                 "constant", constant_values=0)
        self.pred_count.extend([0] * cnt)
        self.label_count.extend([0] * cnt)
        return ids

    def add(self, predicted: List[Input], labels: List[Input]) -> None:
        super().add(predicted, labels)
        predicted = self._convert_ids(predicted)
        labels = self._convert_ids(labels)
        assert self.matrix is not None
        for pred, label in zip(predicted, labels):
            self.matrix[pred, label] += 1
            self.pred_count[pred] += 1
            self.label_count[label] += 1


class ConfusionMatrix(_ConfusionMatrix[Input, Optional[np.ndarray]]):
    def value(self) -> Optional[np.ndarray]:
        return self.matrix


class _MicroMacro(_ConfusionMatrix[Input, float], ABC):
    _valid_modes = ['micro', 'macro', 'macro_weighted']

    def __init__(self, mode: str = 'micro', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        if self.mode not in self._valid_modes:
            raise ValueError(f"Invalid mode {mode}. "
                             f"Supported modes are: {self._valid_modes}")

    def _wrap_micro_macro(self, value: np.ndarray) -> float:
        if self.mode == 'macro':
            value = value.sum() / len(self.class_id)
        elif self.mode == 'macro_weighted':
            value = (value * np.asarray(self.label_count)).sum() / self.count
        return value.item()

    def _true_positive(self) -> np.ndarray:
        value = self.matrix.diagonal()
        return value.sum() if self.mode == 'micro' else value

    def _true_negative(self) -> np.ndarray:
        value = (self.count
                 - np.asarray(self.pred_count)
                 - np.asarray(self.label_count)
                 + self.matrix.diagonal())
        return value.sum() if self.mode == 'micro' else value

    def _false_positive(self) -> int:
        value = np.asarray(self.pred_count) - self.matrix.diagonal()
        return value.sum() if self.mode == 'micro' else value

    def _false_negative(self) -> int:
        value = np.asarray(self.label_count) - self.matrix.diagonal()
        return value.sum() if self.mode == 'micro' else value


class Precision(_MicroMacro[Input]):
    def value(self) -> float:
        return self._wrap_micro_macro(
            self._true_positive() /
            (self._true_positive() + self._false_positive()))


class Recall(_MicroMacro[Input]):
    def value(self) -> float:
        return self._wrap_micro_macro(
            self._true_positive() /
            (self._true_positive() + self._false_negative()))


class F1(Precision[Input], Recall[Input]):
    def value(self) -> float:
        precision = Precision.value(self)
        recall = Recall.value(self)
        return 2 * precision * recall / (precision + recall)
