import functools
import time
from collections import Counter, OrderedDict
from typing import (
    Any, Counter as CounterType, Dict, List, NamedTuple, Optional, Tuple, Type,
    TypeVar, Union)

from mypy_extensions import TypedDict
import torch

from texar.torch.data.data.dataset_utils import Batch
from texar.torch.run.metric import Metric
from texar.torch.utils.types import MaybeList
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
    "TerminateExecution",
    "MetricList",
    "update_metrics",
    "color",
]

T = TypeVar('T')
OptionalList = Optional[MaybeList[T]]
OptionalDict = Optional[Union[T, List[Union[T, Tuple[str, T]]], Dict[str, T]]]
Instance = Union[T, Dict[str, Any]]


def to_list(xs: OptionalList[T]) -> List[T]:
    if isinstance(xs, list):
        return xs
    if xs is None:
        return []
    return [xs]


def to_metric_dict(metrics: OptionalDict[Metric]) \
        -> 'OrderedDict[str, Metric]':
    if metrics is None:
        return OrderedDict()
    if isinstance(metrics, dict):
        if isinstance(metrics, OrderedDict):
            return metrics
        raise ValueError("Metrics must be provided as OrderedDict")
    if isinstance(metrics, list):
        xs = metrics
    else:
        xs = [metrics]
    metric_dict: 'OrderedDict[str, Metric]' = OrderedDict()
    counter: CounterType[str] = Counter()
    for x in xs:
        if isinstance(x, tuple):
            name, x = x
        else:
            name = x.__class__.__name__
        if not isinstance(x, Metric):
            raise ValueError(f"All metrics must be of class Metric, but found "
                             f"{type(x)}")
        if name not in counter:
            metric_dict[name] = x
        else:
            cnt = counter[name]
            if cnt == 1:
                prev_metric = metric_dict[name]
                new_name = f"{name}_{prev_metric.pred_name}"
                if prev_metric.label_name is not None:
                    new_name = f"{new_name}_{prev_metric.label_name}"
                metric_dict[new_name] = prev_metric
                del metric_dict[name]
            new_name = f"{name}_{x.pred_name}"
            if x.label_name is not None:
                new_name = f"{new_name}_{x.label_name}"
            metric_dict[new_name] = x
        counter.update([name])
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
    scheduler: Dict[str, Any]
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

    def __init__(self, size: Optional[int]):
        self.size = size
        self.n_examples = 0
        self.start_time = time.time()

    def add(self, n_examples: int):
        self.n_examples += n_examples

    def progress(self) -> Optional[float]:
        if self.size is None:
            return None
        return self.n_examples / self.size * 100

    def speed(self) -> str:
        speed = self.n_examples / (time.time() - self.start_time)
        if speed > 1.0:
            return f"{speed:.2f}ex/s"
        return f"{1.0 / speed:.2f}s/ex"


class TerminateExecution(Exception):
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
