from abc import ABC
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from texar.torch.run.metric.base_metric import StreamingMetric

__all__ = [
    "Accuracy",
    "ConfusionMatrix",
    "Precision",
    "Recall",
    "F1",
]

Input = TypeVar('Input')
Value = TypeVar('Value')


class Accuracy(StreamingMetric[Input, float]):
    correct: int

    def reset(self) -> None:
        super().reset()
        self.correct = 0

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
        super().add(predicted, labels)
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

    def _convert_ids(self, classes: Sequence[Input]) -> List[int]:
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

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
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
    _valid_modes = ['micro', 'macro', 'weighted']

    def __init__(self, mode: str = 'micro', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        if self.mode not in self._valid_modes:
            raise ValueError(f"Invalid mode {mode}. "
                             f"Supported modes are: {self._valid_modes}")

    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) \
            -> np.ndarray:
        # Credit: sklearn.metrics.classification._prf_divide
        if self.mode == 'micro':
            if denominator == 0.0:
                return np.array(0.0)
            return numerator / denominator

        mask = denominator == 0.0
        denominator = denominator.copy()
        denominator[mask] = 1.0
        value = numerator / denominator
        return value

    def _wrap_micro_macro(self, numerator: np.ndarray,
                          denominator: np.ndarray) -> float:
        value = self._safe_divide(numerator, denominator)
        if self.mode == 'macro':
            value = value.sum() / len(self.class_id)
        elif self.mode == 'weighted':
            value = (value * np.asarray(self.label_count)).sum() / self.count
        return value.item()

    def _true_positive(self) -> np.ndarray:
        assert self.matrix is not None
        value = self.matrix.diagonal()
        return value.sum() if self.mode == 'micro' else value

    def _true_negative(self) -> np.ndarray:
        assert self.matrix is not None
        value = (self.count
                 - np.asarray(self.pred_count)
                 - np.asarray(self.label_count)
                 + self.matrix.diagonal())
        return value.sum() if self.mode == 'micro' else value

    def _false_positive(self) -> int:
        assert self.matrix is not None
        value = np.asarray(self.pred_count) - self.matrix.diagonal()
        return value.sum() if self.mode == 'micro' else value

    def _false_negative(self) -> int:
        assert self.matrix is not None
        value = np.asarray(self.label_count) - self.matrix.diagonal()
        return value.sum() if self.mode == 'micro' else value

    def _value(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        numerator, denominator = self._value()
        return self._wrap_micro_macro(numerator, denominator)


class Precision(_MicroMacro[Input]):
    def _value(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._true_positive(),
                (self._true_positive() + self._false_positive()))


class Recall(_MicroMacro[Input]):
    def _value(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._true_positive(),
                (self._true_positive() + self._false_negative()))


class F1(Precision[Input], Recall[Input]):
    def _value(self) -> Tuple[np.ndarray, np.ndarray]:
        # pylint: disable=protected-access
        precision = self._safe_divide(*Precision._value(self))
        recall = self._safe_divide(*Recall._value(self))
        # pylint: enable=protected-access
        return (2 * precision * recall,
                precision + recall)
