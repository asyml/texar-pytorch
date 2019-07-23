import os
import random
import warnings
from typing import Any, Dict, IO, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer

from texar.torch.data.data.data_base import DataBase
from texar.torch.data.data.data_iterators import BatchingStrategy, DataIterator
from texar.torch.data.data.dataset_utils import Batch
from texar.torch.run.action import Action
from texar.torch.run.condition import Condition, Event, HookPoint
from texar.torch.run.metric import Metric
from texar.torch.utils.types import MaybeList
from texar.torch.utils.utils import get_instance

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
OptionalList = Optional[MaybeList[T]]
Instance = Union[T, Dict[str, Any]]


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


def _to_instance(typ: Type[T], instance: Instance[T],
                 modules: List[str]) -> Optional[T]:
    if instance is None:
        return None
    if isinstance(instance, dict):
        instance = get_instance(instance['type'], instance['kwargs'], modules)
    if not isinstance(instance, typ):
        raise ValueError(
            f"The instance {instance} is not of type {typ.__name__}")
    return instance


class TrainState(NamedTuple):
    model: Dict[str, torch.Tensor]
    optimizer: Dict[str, torch.Tensor]
    scheduler: Dict[str, Any]
    system_rng: Any
    numpy_rng: Any
    torch_rng: Any


class Executor:
    _defaults = {
        "log_format": "{date} {time} : Epoch {epoch} @ {iteration}it "
                      "({progress}%), loss = {loss:.3f}",
        "eval_log_format": "{date} {time} : Epoch {epoch}, "
                           "{split} result = {{{metrics:.3f}}}",
    }

    # TODO: Add a unified `state` that is readonly and keeps track of
    #   everything. We'll only have to pass it to hooks.

    def __init__(self, model: nn.Module,
                 train_data: Optional[DataBase] = None,
                 *,
                 valid_data: OptionalList[DataBase] = None,
                 test_data: OptionalList[DataBase] = None,
                 batching_strategy: Optional[BatchingStrategy] = None,
                 device: Optional[torch.device] = None,
                 # Checkpoint
                 checkpoint_dir: Optional[str] = None,
                 max_to_keep: Optional[int] = None,
                 save_every: OptionalList[Condition] = None,
                 save_training_state: bool = True,  # whether to save optimizer, scheduler, RNG state
                 # Training
                 train_metrics: Optional[Dict[str, Metric]] = None,
                 optimizer: Optional[Instance[Optimizer]] = None,
                 lr_scheduler: Optional[Instance[LRScheduler]] = None,
                 max_epochs: Optional[int] = None,
                 num_backwards_per_update: int = 1,
                 grad_clip: Optional[float] = None,
                 # Validation
                 valid_metrics: Optional[Dict[str, Metric]] = None,
                 validate_every: OptionalList[Condition] = None,
                 early_stop_patience: Optional[int] = None,
                 plateau_condition: OptionalList[Condition] = None,
                 action_on_plateau: OptionalList[Action] = None,
                 validate_mode: str = 'eval',
                 # Testing
                 test_every: OptionalList[Condition] = None,
                 test_mode: str = 'predict',
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
        self.batching_strategy = batching_strategy
        self._moved_model = False

        if device is None:
            if torch.cuda.is_available():
                device = torch.device(torch.cuda.current_device())
            else:
                device = torch.device('cpu')
        self.device = device

        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.save_conditions = _to_list(save_every)
        self.save_training_state = save_training_state

        self.train_metrics = _to_dict(train_metrics)
        self.optimizer = _to_instance(
            Optimizer, optimizer, ["torch.optim", "texar.core"])
        self.lr_scheduler = _to_instance(
            LRScheduler, lr_scheduler,
            ["torch.optim.lr_scheduler", "texar.core"])
        self.max_epochs = max_epochs
        self.num_backwards_per_update = num_backwards_per_update
        self.grad_clip = grad_clip

        self.valid_metrics = _to_dict(valid_metrics)
        self.valid_conditions = _to_list(validate_every)
        self.valid_mode = validate_mode

        self.test_conditions = _to_list(test_every)
        self.test_mode = test_mode

        self.log_conditions = _to_list(log_every)
        self.log_format: str
        self.log_destination: List[Union[str, IO[str]]]
        self.eval_log_format: str

        for attr, default in self._defaults.items():
            value = locals()[attr]
            if value is None:
                value = default
            setattr(self, attr, value)

        # Initialize hooks.
        self.hooks: Dict[HookPoint, List[Tuple[Condition, Action]]] = {
            (event, point): []
            for event in Event for point in ['begin', 'end']}

        # Detect event-action cycles.
        # TODO: Maybe don't do this.

        # Validate arguments.

    def _register_hook(self, event: Event, point: str, action: Action,
                       cond: Optional[Condition] = None):
        if (event, point) not in self.hooks:
            raise ValueError(f"Invalid hook point ({event}, {point})")
        self.hooks[(event, point)].append((cond, action))

    def on(self, event: Event, point: str = 'end'):
        if (event, point) not in self.hooks:
            raise ValueError(f"Invalid hook point ({event}, {point})")

        def wrapper(func):
            self.hooks[(event, point)].append((None, func))
            return func

        return wrapper

    def _fire_event(self, event: Event, point: str, *args, **kwargs):
        for cond, action in self.hooks[(event, point)]:
            if cond is None or cond._hooks[(event, point)]():
                action(*args, **kwargs)

    def _validate_step(self):
        pass

    def _test_step(self):
        pass

    def _train_step(self, batch: Batch, epoch: int, iteration: int):
        return_dict = self.model(batch)
        try:
            loss = return_dict['loss']
        except KeyError:
            raise ValueError("Return dictionary from model does not "
                             "contain 'loss' entry")
        loss.backward()
        if iteration % self.num_backwards_per_update == 0:
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return return_dict

    def _save(self):
        # TODO: Save a meta file?

        if self.save_training_state and self.optimizer is not None:
            train_state = TrainState(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=(self.lr_scheduler.state_dict()
                           if self.lr_scheduler is not None else None),
                system_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                torch_rng=torch.random.get_rng_state(),
            )
            torch.save(train_state, path)
        else:
            torch.save(self.model.state_dict(), path)

    def load(self, path: Optional[str] = None,
             load_training_state: bool = True):
        r"""Load a previous model checkpoint from file.

        Args:
            path (str, optional): Path to a specific checkpoint or a checkpoint
                directory. If a directory is specified, the most recent
                checkpoint in the directory is loaded. If `None`,
                :attr:`checkpoint_dir` will be used.
            load_training_state (bool): If `True`, will load entire training
                state from checkpoint (if the checkpoint contains training
                state). Otherwise, just load model weights. Defaults to `True`.
        """
        if path is None and self.checkpoint_dir is None:
            raise ValueError("Path must be specified when `checkpoint_dir` is")
        path = self.checkpoint_dir if path is None else path
        if os.path.isdir(path):
            files = [os.path.join(path, name) for name in os.listdir(path)
                     if name.endswith('.pt')]
            path = max(files, key=os.path.getmtime)

        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, TrainState):
            self.model.load_state_dict(checkpoint.model)
            if load_training_state:
                self.optimizer.load_state_dict(checkpoint.optimizer)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint.scheduler)
                random.setstate(checkpoint.system_rng)
                np.random.set_state(checkpoint.numpy_rng)
                torch.random.set_rng_state(checkpoint.torch_rng)
        else:
            self.model.load_state_dict(checkpoint)

    def train(self, dataset: Optional[DataBase] = None):
        if self.max_epochs is None and early_stop_patience is None:
            warnings.warn("Neither `max_epochs` nor early stopping is "
                          "configured. Training will run indefinitely.")

        if self.optimizer is None:
            raise ValueError("Optimizer is not specified")
        self.optimizer.zero_grad()

        if dataset is None and self.train_data is None:
            raise ValueError("No training dataset is specified")
        dataset = self.train_data if dataset is None else dataset
        dataset.to(self.device)
        iterator = DataIterator(dataset, self.batching_strategy)

        if not self._moved_model:
            self.model.to(self.device)

        iteration = 0
        self._fire_event(Event.Training, 'begin')
        for epoch in range(1, self.max_epochs + 1):
            self._fire_event(Event.Epoch, 'begin')
            for batch in iterator:
                self._fire_event(Event.Iteration, 'begin')
                iteration += 1
                self._train_step(batch, epoch, iteration)
                self._fire_event(Event.Iteration, 'end', iteration)
            self._fire_event(Event.Epoch, 'end', epoch)
        self._fire_event(Event.Training, 'end')

    def validate(self, dataset: Optional[DataBase] = None):
        pass

    def test(self, dataset: Optional[DataBase] = None):
        pass
