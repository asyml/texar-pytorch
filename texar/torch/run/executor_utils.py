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
Utility functions for the Executor module.
"""

import functools
import time
from collections import Counter, OrderedDict
from typing import (
    Any, Callable, Counter as CounterType, Dict, Iterator, List, NamedTuple,
    Optional, Tuple, Type, TypeVar, Union, Mapping, Sequence)

from mypy_extensions import TypedDict
import torch
from torch import nn

from texar.torch.data.data.dataset_utils import Batch
from texar.torch.run.metric import Metric
from texar.torch.utils.types import MaybeSeq
from texar.torch.utils.utils import get_instance

__all__ = [
    "OptionalList",
    "OptionalDict",
    "Instance",
    "to_list",
    "to_metric_dict",
    "to_instance",
    "SavedTrainingState",
    "TrainingStatus",
    "CheckpointMetaInfo",
    "ProgressTracker",
    "ExecutorTerminateSignal",
    "MetricList",
    "update_metrics",
    "color",
    "repr_module",
]

T = TypeVar('T')
OptionalList = Optional[MaybeSeq[T]]
OptionalDict = Optional[Union[T, Sequence[Union[T, Tuple[str, T]]],
                              Mapping[str, T]]]
Instance = Union[T, Dict[str, Any]]


def to_list(xs: OptionalList[T]) -> List[T]:
    if isinstance(xs, Sequence):
        return list(xs)
    if xs is None:
        return []
    return [xs]


def _to_dict(ds: OptionalDict[T],
             unambiguous_name_fn: Callable[[str, T, int], str],
             default_name_fn: Callable[[int, T], str]) -> 'OrderedDict[str, T]':
    if ds is None:
        return OrderedDict()
    if isinstance(ds, Mapping):
        return OrderedDict(ds)
    if isinstance(ds, Sequence):
        xs = ds
    else:
        xs = [ds]
    ret_dict: 'OrderedDict[str, T]' = OrderedDict()
    counter: CounterType[str] = Counter()
    for idx, item in enumerate(xs):
        if isinstance(item, tuple):
            name, item = item
        else:
            name = default_name_fn(idx, item)
        if name not in counter:
            ret_dict[name] = item
        else:
            cnt = counter[name]
            if cnt == 1:
                prev_item = ret_dict[name]
                ret_dict[unambiguous_name_fn(name, prev_item, 1)] = prev_item
                del ret_dict[name]
            ret_dict[unambiguous_name_fn(name, item, cnt + 1)] = item
        counter.update([name])
    return ret_dict


def to_dict(xs: OptionalDict[T],
            default_name: Optional[str] = None) -> Dict[str, T]:
    def unambiguous_name_fn(name: str, _item: T, cnt: int) -> str:
        return f"{name}.{cnt}"

    def default_name_fn(idx: int, _item: T) -> str:
        if default_name is not None:
            return default_name
        return str(idx)

    return _to_dict(xs, unambiguous_name_fn, default_name_fn)


def to_metric_dict(metrics: OptionalDict[Metric]) -> 'OrderedDict[str, Metric]':
    def unambiguous_name_fn(name: str, metric: Metric, _cnt: int) -> str:
        new_name = f"{name}_{metric.pred_name}"
        if metric.label_name is not None:
            new_name = f"{new_name}_{metric.label_name}"
        return new_name

    def default_name_fn(_idx: int, metric: Metric) -> str:
        return metric.metric_name

    if isinstance(metrics, dict) and not isinstance(metrics, OrderedDict):
        raise ValueError("Metrics dictionary must be of type OrderedDict")
    metric_dict = _to_dict(metrics, unambiguous_name_fn, default_name_fn)

    for name, metric in metric_dict.items():
        if not name.isidentifier():
            raise ValueError(f"Name \"{name}\" for metric {metric} is not a "
                             f"valid identifier name")
        if not isinstance(metric, Metric):
            raise ValueError(f"All metrics must be of class Metric, but found "
                             f"{type(metric)}")
    return metric_dict


def to_instance(typ: Type[T], instance: Instance[T], modules: List[str],
                extra_kwargs: Optional[Dict[str, Any]] = None) -> Optional[T]:
    if instance is None:
        return None
    if isinstance(instance, dict):
        kwargs = {**instance.get('kwargs', {}), **(extra_kwargs or {})}
        instance = get_instance(instance['type'], kwargs, modules)
    if not isinstance(instance, typ):
        raise ValueError(f"The instance {instance} is not of type {typ}")
    return instance


# TODO: Also save training progress?
class SavedTrainingState(NamedTuple):
    r"""The entire training state to save to or load from checkpoints."""
    model: Dict[str, torch.Tensor]
    optimizer: Dict[str, torch.Tensor]
    scheduler: Optional[Dict[str, Any]]
    system_rng: Any
    numpy_rng: Any
    torch_rng: Any


class TrainingStatus(TypedDict):
    epoch: int
    iteration: int
    split: str
    metric: 'OrderedDict[str, Metric]'
    eval_metric: 'OrderedDict[str, Metric]'


class CheckpointMetaInfo(TypedDict):
    status: TrainingStatus
    timestamp: float


class ProgressTracker:
    start_time: float
    size: Optional[int]
    n_examples: int
    accumulated_time: float
    paused: bool

    _tracker_stack: List['ProgressTracker'] = []

    def __init__(self, size: Optional[int] = None):
        self.size = size
        self.started = False

    def set_size(self, size: Optional[int]):
        self.size = size

    def start(self):
        if self.started:
            return
        self.started = True
        if len(self._tracker_stack) > 0:
            self._tracker_stack[-1].pause()
        self._tracker_stack.append(self)
        self.reset()
        self.paused = False

    def stop(self):
        if not self.started:
            return
        self.started = False
        obj = self._tracker_stack.pop(-1)
        assert obj is self
        if len(self._tracker_stack) > 0:
            self._tracker_stack[-1].resume()

    def reset(self):
        self.n_examples = 0
        self.start_time = time.time()
        self.accumulated_time = 0.0

    def pause(self):
        if self.paused:
            return
        self.paused = True
        self.accumulated_time += time.time() - self.start_time

    def resume(self):
        if not self.paused:
            return
        self.paused = False
        self.start_time = time.time()

    def add(self, n_examples: int):
        self.n_examples += n_examples

    def progress(self) -> Optional[float]:
        if self.size is None:
            return None
        return self.n_examples / self.size * 100

    def time_elapsed(self) -> float:
        return self.accumulated_time + (time.time() - self.start_time)

    def speed(self) -> str:
        speed = self.n_examples / self.time_elapsed()
        if speed > 1.0:
            return f"{speed:.2f}ex/s"
        try:
            return f"{1.0 / speed:.2f}s/ex"
        except ZeroDivisionError:
            return f"0.00ex/s"


class ExecutorTerminateSignal(Exception):
    pass


@functools.total_ordering
class MetricList:
    r"""A class representing list of metrics along with their values at a
    certain point. Used for metric comparisons.

    Args:
        metrics: The dictionary of metric instances.
        values (optional): The dictionary of metric values. If `None` (default),
            the current values of the provided metrics are used.
    """

    # TODO: Ignore non-streaming metrics here? Or in

    def __init__(self, metrics: 'OrderedDict[str, Metric]',
                 values: Optional[Dict[str, Any]] = None):
        self.metrics = metrics
        if values is None:
            self.values = {name: metric.value()
                           for name, metric in metrics.items()}
        else:
            self.values = values

    def _compare_metrics(self, other: Any):
        if not isinstance(other, MetricList):
            raise ValueError(
                "Cannot compare to an object not of type MetricList")
        for (name, metric), (other_name, other_metric) in zip(
                self.metrics.items(), other.metrics.items()):
            if name != other_name or type(metric) is not type(other_metric):
                raise ValueError("Cannot compare two metric lists with "
                                 "different base metrics")

    def __eq__(self, other: Any) -> bool:
        self._compare_metrics(other)
        return all(self.values[name] == other.values[name]
                   for name in self.metrics)

    def __gt__(self, other: 'MetricList') -> bool:
        r"""Compare this metric list to another, and return whether the current
        list is better.
        """
        self._compare_metrics(other)
        for name, metric in self.metrics.items():
            cmp = metric.better(self.values[name], other.values[name])
            if cmp is not None:
                return cmp
        return False


def update_metrics(return_dict: Dict[str, Any], batch: Batch,
                   metrics: 'OrderedDict[str, Metric]') -> None:
    for metric_name, metric in metrics.items():
        if metric.pred_name is not None:
            try:
                pred_val = return_dict[metric.pred_name]
            except KeyError:
                raise ValueError(
                    f"Return dictionary from model does not contain "
                    f"'{metric.pred_name}' entry, which was required for "
                    f"metric '{metric_name}'")
            if isinstance(pred_val, torch.Tensor):
                pred_val = pred_val.tolist()
            pred_val = to_list(pred_val)
        else:
            pred_val = None
        if metric.label_name is not None:
            try:
                label_val = batch[metric.label_name]
            except KeyError:
                raise ValueError(
                    f"Data batch does not contain '{metric.label_name}' "
                    f"entry, which was required for metric '{metric_name}'")
            if isinstance(label_val, torch.Tensor):
                label_val = label_val.tolist()
            label_val = to_list(label_val)
        else:
            label_val = None
        metric.add(pred_val, label_val)


CLEAR_LINE = '\033[2K\r'
RESET_CODE = '\033[0m'
COLOR_CODE = {
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[94m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'gray': '\033[37m',
    'grey': '\033[37m'
}


def color(s: str, col: str):
    return COLOR_CODE[col.lower()] + s + RESET_CODE


def _add_indent(s_, n_spaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(n_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def _convert_id(keys: List[str]) -> Iterator[str]:
    start = end = None
    for key in keys:
        if key.isnumeric() and end == int(key) - 1:
            end = int(key)
        else:
            if start is not None:
                if start == end:
                    yield f"id {start}"
                else:
                    yield f"ids {start}-{end}"
            if key.isnumeric():
                start = end = int(key)
            else:
                start = end = None
                yield key
    if start is not None:
        if start == end:
            yield f"id {start}"
        else:
            yield f"ids {start}-{end}"


def repr_module(module: nn.Module) -> str:
    r"""Create a compressed representation by combining identical modules in
    `nn.ModuleList`s and `nn.ParameterList`s.
    """

    # We treat the extra repr like the sub-module, one item per line
    extra_lines: List[str] = []
    extra_repr = module.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    prev_mod_str = None
    keys: List[str] = []
    for key, submodule in module.named_children():
        mod_str = repr_module(submodule)
        mod_str = _add_indent(mod_str, 2)
        if prev_mod_str is None or prev_mod_str != mod_str:
            if prev_mod_str is not None:
                for name in _convert_id(keys):
                    child_lines.append(f"({name}): {prev_mod_str}")
            prev_mod_str = mod_str
            keys = [key]
        else:
            keys.append(key)
    if len(keys) > 0:
        for name in _convert_id(keys):
            child_lines.append(f"({name}): {prev_mod_str}")
    lines = extra_lines + child_lines

    main_str = module.__class__.__name__ + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str
