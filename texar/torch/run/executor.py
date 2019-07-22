import warnings
from typing import Any, Dict, IO, List, Optional, TypeVar, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer

from texar.torch.data.data.data_base import DataBase
from texar.torch.run.action import Action
from texar.torch.run.condition import Condition, Event
from texar.torch.run.metric import Metric
from texar.torch.utils.types import MaybeList
from texar.torch.utils.utils import get_instance

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
OptionalList = Optional[MaybeList[T]]
Instance = Union[T, Dict[str, Any]]

_defaults = {
    "log_format": "{date} {time} : Epoch {epoch} @ {iteration}it "
                  "({progress}%), loss = {loss:.3f}",
    "eval_log_format": "{date} {time} : Epoch {epoch}, "
                       "{split} result = {{{metrics:.3f}}}",
}


def _to_list(xs: OptionalList[T]) -> List[T]:
    if xs is None:
        return []
    if isinstance(xs, list):
        return xs
    return [xs]


def _to_dict(d: Optional[Dict[K, V]]) -> Dict[K, V]:
    if d is None:
        return {}
    return d


def _to_instance(instance: Instance[T], modules: List[str]) -> Optional[T]:
    if instance is None:
        return None
    if isinstance(instance, dict):
        instance = get_instance(instance['type'], instance['kwargs'], modules)
    return instance


class Executor:
    def __init__(self, model: nn.Module,
                 train_data: Optional[DataBase] = None,
                 valid_data: OptionalList[DataBase] = None,
                 test_data: OptionalList[DataBase] = None,
                 device: Optional[torch.device] = None,
                 # Checkpoint
                 checkpoint_dir: Optional[str] = None,
                 max_to_keep: Optional[int] = None,
                 save_every: OptionalList[Condition] = None,
                 # Training
                 train_metrics: Optional[Dict[str, Metric]] = None,
                 optimizer: Optional[Instance[Optimizer]] = None,
                 lr_scheduler: Optional[Instance[LRScheduler]] = None,
                 max_epochs: Optional[int] = None,
                 # Validation
                 valid_metrics: Optional[Dict[str, Metric]] = None,
                 validate_every: OptionalList[Condition] = None,
                 early_stop_patience: Optional[int] = None,
                 plateau_condition: OptionalList[Condition] = None,
                 action_on_plateau: OptionalList[Action] = None,
                 # Testing
                 test_every: OptionalList[Condition] = None,
                 # Logging
                 log_every: OptionalList[Condition] = None,
                 log_format: Optional[str] = None,
                 log_destination: OptionalList[Union[str, IO[str]]] = None,
                 eval_log_format: Optional[str] = None,
                 # Tensorboard
                 tensorboard_log_dir: Optional[str] = None,
                 write_summary_every: OptionalList[Condition] = None):
        self.model = model
        self.train_data = train_data
        self.valid_data = _to_list(valid_data)
        self.test_data = _to_list(test_data)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device(torch.cuda.current_device())
            else:
                device = torch.device('cpu')
        self.device = device

        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.max_epochs = max_epochs

        self.train_metrics = _to_dict(train_metrics)
        self.optimizer = _to_instance(optimizer, ["torch.optim", "texar.core"])
        self.lr_scheduler = _to_instance(
            lr_scheduler, ["torch.optim.lr_scheduler", "texar.core"])

        self.valid_metrics = _to_dict(valid_metrics)
        self.valid_conditions = _to_list(validate_every)

        self.test_conditions = _to_list(test_every)

        self.log_conditions = _to_list(log_every)
        self.log_format: str
        self.log_destination: List[Union[str, IO[str]]]
        self.eval_log_format: str

        for attr, default in _defaults.items():
            value = locals()[attr]
            if value is None:
                value = default
            setattr(self, attr, value)

        # Initialize hooks.
        self.hooks = {(event.value, point): []
                      for event in Event for point in ['begin', 'end']}

        # Detect event-action cycles.

    def on(self, event: str, point: str = 'end'):
        # return decorator
        pass

    def _train_step(self):
        pass

    def _validate_step(self):
        pass

    def _test_step(self):
        pass

    def train(self):
        if self.max_epochs is None and early_stop_patience is None:
            warnings.warn("Neither `max_epochs` nor early stopping is "
                          "configured. Training will run indefinitely.")

    def validate(self):
        pass

    def test(self):
        pass
