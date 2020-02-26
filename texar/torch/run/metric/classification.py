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
Executor metrics for classification tasks.
"""

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
    r"""The accuracy metric for evaluation classification tasks. Accuracy is
    defined as the ratio of correct (exactly matching) predictions out of all
    predictions.

    Accuracy is a :class:`~texar.torch.run.metric.StreamingMetric`, requires
    both predicted values and labels. Accuracy values are :class:`float`
    numbers between 0 and 1, with higher values being better.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
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
    _class_id: Dict[Input, int]

    def reset(self) -> None:
        super().reset()
        self.matrix = None
        self.pred_count = []
        self.label_count = []
        self._class_id = {}

    def _convert_ids(self, classes: Sequence[Input]) -> List[int]:
        ids = []
        cnt = 0
        for klass in classes:
            if klass not in self._class_id:
                self._class_id[klass] = len(self._class_id)
                cnt += 1
            ids.append(self._class_id[klass])
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
    r"""The confusion matrix is an evaluation metric for classification tasks.

    Confusion matrix is a :class:`~texar.torch.run.metric.StreamingMetric`,
    requires both predicted values and labels. Confusion matrix values are NumPy
    arrays, with no clear definition of "better". Comparison of two confusion
    matrices are not meaningful.

    The value indexed at ``(i, j)`` of the confusion matrix is the number of
    data points whose predicted label is `i` and whose ground truth label is
    `j`. Labels are internally mapped to indices.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def value(self) -> Optional[np.ndarray]:
        return self.matrix

    @property
    def class_id(self):
        r"""Mapping of predicted values and labels to indices within the matrix.
        """
        return self._class_id

    def better(self, cur: Value, prev: Value) -> Optional[bool]:
        # Always return `None` to indicate values are uncomparable.
        return None


class _MicroMacro(_ConfusionMatrix[Input, float], ABC):
    _valid_modes = ['binary', 'micro', 'macro', 'weighted']

    def __init__(self, mode: str = 'binary', pos_label: Optional[Input] = None,
                 *, pred_name: str, label_name: str = "label"):
        super().__init__(pred_name=pred_name, label_name=label_name)
        self.mode = mode
        if self.mode not in self._valid_modes:
            raise ValueError(f"Invalid mode {mode}. "
                             f"Supported modes are: {self._valid_modes}")
        if self.mode == 'binary' and pos_label is None:
            raise ValueError("`pos_label` must not be none when `mode` is "
                             "set to 'binary'")
        if pos_label is not None:
            self.pos_label = pos_label

    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) \
            -> np.ndarray:
        # Credit: sklearn.metrics.classification._prf_divide
        if numerator.size == 1:
            if denominator == 0.0:
                return np.array(0.0)
            return numerator / denominator

        mask = denominator == 0.0
        denominator = denominator.copy()
        denominator[mask] = 1.0
        value = numerator / denominator
        return value

    def _convert_value(self, value: np.ndarray) -> np.ndarray:
        if self.mode == 'binary':
            label = self._class_id.get(self.pos_label, None)
            if label is None:
                return np.array(0)
            return value[label]
        if self.mode == 'micro':
            return value.sum()
        return value

    def _true_positive(self) -> np.ndarray:
        assert self.matrix is not None
        value = self.matrix.diagonal()
        return self._convert_value(value)

    def _true_negative(self) -> np.ndarray:
        assert self.matrix is not None
        value = (self.count
                 - np.asarray(self.pred_count)
                 - np.asarray(self.label_count)
                 + self.matrix.diagonal())
        return self._convert_value(value)

    def _false_positive(self) -> np.ndarray:
        assert self.matrix is not None
        value = np.asarray(self.pred_count) - self.matrix.diagonal()
        return self._convert_value(value)

    def _false_negative(self) -> np.ndarray:
        assert self.matrix is not None
        value = np.asarray(self.label_count) - self.matrix.diagonal()
        return self._convert_value(value)

    def _value(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return the numerator and denominator of the metric value.
        """
        raise NotImplementedError

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        numerator, denominator = self._value()
        value = self._safe_divide(numerator, denominator)
        if self.mode == 'macro':
            value = value.sum() / len(self._class_id)
        elif self.mode == 'weighted':
            value = (value * np.asarray(self.label_count)).sum() / self.count
        return value.item()


class Precision(_MicroMacro[Input]):
    r"""The precision metric for evaluation classification tasks. Precision is
    defined as the ratio of ``tp / (tp + fp)``, where ``tp`` is the number of
    true positives and ``fp`` is the number of false positives.

    Precision is a :class:`~texar.torch.run.metric.StreamingMetric`, requires
    both predicted values and labels. Precision values are :class:`float`
    numbers between 0 and 1, with higher values being better.

    Args:
        mode (str): The mode for computing averages across multiple labels.
            Defaults to ``"binary"``. Available options include:

            - ``"binary"``: Only report results for the class specified by
              :attr:`pos_label`. This is only meaningful for binary
              classification tasks.
            - ``"micro"``: Return the precision value computed using globally
              counted true positives and false positives.
            - ``"macro"``: Return the unweighted average of precision values for
              each label.
            - ``"weighted"``: Return the average of precision values for each
              label, weighted by the number of true instances for each label.
        pos_label (str, optional): The label for the positive class. Only used
            if :attr:`mode` is set to ``"binary"``.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def _value(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._true_positive(),
                (self._true_positive() + self._false_positive()))


class Recall(_MicroMacro[Input]):
    r"""The recall metric for evaluation classification tasks. Precision is
    defined as the ratio of ``tp / (tp + fn)``, where ``tp`` is the number of
    true positives and ``fn`` is the number of false negatives.

    Recall is a :class:`~texar.torch.run.metric.StreamingMetric`, requires both
    predicted values and labels. Recall values are :class:`float` numbers
    between 0 and 1, with higher values being better.

    Args:
        mode (str): The mode for computing averages across multiple labels.
            Defaults to ``"binary"``. Available options include:

            - ``"binary"``: Only report results for the class specified by
              :attr:`pos_label`. This is only meaningful for binary
              classification tasks.
            - ``"micro"``: Return the recall value computed using globally
              counted true positives and false negatives.
            - ``"macro"``: Return the unweighted average of recall values for
              each label.
            - ``"weighted"``: Return the average of recall values for each
              label, weighted by the number of true instances for each label.
        pos_label (str, optional): The label for the positive class. Only used
            if :attr:`mode` is set to ``"binary"``.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def _value(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._true_positive(),
                (self._true_positive() + self._false_negative()))


class F1(Precision[Input], Recall[Input]):
    r"""The F1 metric for evaluation classification tasks. F1 is defined as the
    harmonic mean of precision and recall.

    F1 is a :class:`~texar.torch.run.metric.StreamingMetric`, requires both
    predicted values and labels. F1 values are :class:`float` numbers between 0
    and 1, with higher values being better.

    Args:
        mode (str): The mode for computing averages across multiple labels.
            Defaults to ``"binary"``. Available options include:

            - ``"binary"``: Only report results for the class specified by
              :attr:`pos_label`. This is only meaningful for binary
              classification tasks.
            - ``"micro"``: Return the F1 value computed using globally counted
              true positives, false positives, and false negatives.
            - ``"macro"``: Return the unweighted average of F1 values for each
              label.
            - ``"weighted"``: Return the average of F1 values for each label,
              weighted by the number of true instances for each label.
        pos_label (str, optional): The label for the positive class. Only used
            if :attr:`mode` is set to ``"binary"``.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def _value(self) -> Tuple[np.ndarray, np.ndarray]:
        # pylint: disable=protected-access
        precision = self._safe_divide(*Precision._value(self))
        recall = self._safe_divide(*Recall._value(self))
        # pylint: enable=protected-access
        return (2 * precision * recall,
                precision + recall)
