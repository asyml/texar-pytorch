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
Base classes for Executor metrics.
"""

import sys
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Sequence, TYPE_CHECKING, TypeVar

__all__ = [
    "Metric",
    "SimpleMetric",
    "StreamingMetric",
]

Input = TypeVar('Input')
Value = TypeVar('Value')

if not TYPE_CHECKING and sys.version_info[:2] <= (3, 6):
    # In Python 3.6 and below, pickling a `Generic` subclass that is specialized
    # would cause an exception. To prevent troubles with `Executor` save & load,
    # we use a dummy implementation of `Generic` through our home-brew
    # `GenericMeta`.
    from abc import ABCMeta  # pylint: disable=ungrouped-imports

    class GenericMeta(ABCMeta):
        def __getitem__(cls, params):
            # Whatever the parameters, just return the same class.
            return cls

    class Generic(metaclass=GenericMeta):  # pylint: disable=function-redefined
        pass


class Metric(Generic[Input, Value], ABC):
    r"""Base class of all metrics. You should not directly inherit this class,
    but inherit from :class:`SimpleMetric` or :class:`StreamingMetric` instead.

    Subclasses can override the class attributes to indicate their behaviors:

    - :attr:`higher_is_better`: If `True`, higher (comparison using the greater
      than operator ``>`` returns `True`) values are considered better metric
      values. If `False`, lower values are considered better. Defaults to
      `True`.
    - :attr:`required_pred`: If `True`, predicted values are required to compute
      the metric value. Defaults to `True`.
    - :attr:`requires_label`: If `True`, labels are required to compute the
      metric value. Defaults to `True`.

    Keyword Args:
        pred_name (str, optional): Name of the predicted value. This will be
            used as the key to the dictionary returned by the model.
        label_name (str, optional): Name of the label. This will be used as the
            key to the batch object returned by the dataset. Defaults to
            ``"label"``.
        higher_is_better (bool, optional): If specified, the
            :attr:`higher_is_better` attribute for the instance is overwritten
            by the specified value.
    """
    higher_is_better: bool = True
    requires_pred: bool = True
    requires_label: bool = True

    def __init__(self, *, pred_name: Optional[str],
                 label_name: Optional[str] = "label",
                 higher_is_better: Optional[bool] = None):
        self.reset()
        if self.requires_pred and pred_name is None:
            raise ValueError(f"Metric {self.metric_name} requires a "
                             f"prediction name, but None is provided")
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
        r"""Name of the metric. By default, the class name is used."""
        return self.__class__.__name__

    @property
    def pred_name(self) -> Optional[str]:
        r"""Name of the predicted value. This will be used as the key to the
        dictionary returned by the model.
        """
        return self._pred_name

    @property
    def label_name(self) -> Optional[str]:
        r"""Name of the label (ground truth / gold value). This will be used as
        the key to the batch object returned by the dataset.
        """
        return self._label_name

    @abstractmethod
    def reset(self) -> None:
        r"""Reset the internal state of the metric, and erase all previously
        added data points.
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
        r"""Record a data batch in the metric.

        Args:
            predicted: The list of predicted values.
            labels: The list of labels.
        """
        raise NotImplementedError

    @abstractmethod
    def value(self) -> Value:
        r"""Compute the metric value.

        Returns:
            The metric value.
        """
        raise NotImplementedError

    def better(self, cur: Value, prev: Value) -> Optional[bool]:
        r"""Compare two metric values and return which is better.

        Args:
            cur: The "current" metric value.
            prev: The "previous" metric value.

        Returns:
            Return value is either a `bool` or `None`.

            - If `True`, the current metric value is considered better.
            - If `False`, the previous metric value is considered better.
            - If `None`, the two values are considered to be the same, or
              uncomparable.
        """
        result = (True if cur > prev else  # type: ignore
                  False if cur < prev else None)  # type: ignore
        if not self.higher_is_better and result is not None:
            result = not result
        return result

    def finalize(self, executor) -> None:
        r"""Finalize the metric. Called when the whole dataset has been fully
        iterated, e.g., at the end of an epoch, or the end of validation or
        testing.

        The default behavior is no-op. Most metrics won't need this, special
        ones such as :class:`FileWriterMetric` utilizes this to performs
        one-time only operations.

        Args:
            executor: The :class:`Executor` instance.
        """


class SimpleMetric(Metric[Input, Value], ABC):
    r"""Base class of simple metrics. Simple metrics are metrics that do not
    support incremental computation. The value of the metric is computed only
    after all data points have been added.

    The default implementation of :meth:`add` simply stores the predicted values
    and labels into lists.
    """
    labels: List[Input]
    predicted: List[Input]
    _cached_value: Optional[Value]

    def reset(self) -> None:
        self.labels = []
        self.predicted = []
        self._cached_value = None

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]):
        if (self.requires_pred and self.requires_label and
                len(predicted) != len(labels)):
            raise ValueError(
                "Lists `predicted` and `labels` should have the same length")
        if self.requires_pred:
            self.predicted.extend(predicted)
        if self.requires_label:
            self.labels.extend(labels)
        self._cached_value = None

    def value(self):
        if self._cached_value is not None:
            return self._cached_value
        self._cached_value = self._value()
        return self._cached_value

    def _value(self) -> Value:
        r"""Compute the metric value. This function is called in
        :meth:`texar.torch.run.metric.SimpleMetric.value` and the output is
        cached. This prevents recalculation of metrics which may be time
        consuming.

        Returns:
            The metric value.
        """
        raise NotImplementedError


class StreamingMetric(Metric[Input, Value], ABC):
    r"""Base class of streaming metrics. Streaming metrics are metrics that
    support incremental computation. The value of the metric may be queried
    before all data points have been added, and the computation should not be
    expensive.

    The default implementation of :meth:`add` only keeps track of the number of
    data points added. You should override this method.
    """
    count: int

    def reset(self) -> None:
        self.count = 0

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
        if len(predicted) != len(labels):
            raise ValueError(
                "Lists `predicted` and `labels` should have the same length")
        self.count += len(predicted)
