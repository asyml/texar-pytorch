import functools
import os
import random
import warnings
from collections import defaultdict, Counter
from typing import (Any, Callable, Counter as CounterType, Dict, IO, List,
                    NamedTuple, Optional, Type, TypeVar, Union, overload)

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

__all__ = [
    "make_deterministic",
    "Executor",
]

T = TypeVar('T')
OptionalList = Optional[MaybeList[T]]
OptionalDict = Optional[Union[T, List[T], Dict[str, T]]]
Instance = Union[T, Dict[str, Any]]
HookFn = Callable[..., None]


def make_deterministic(seed: int = 19260817,
                       cudnn_deterministic: bool = False):
    r"""Make experiment deterministic by using specific random seeds across
    all frameworks and (optionally) use deterministic algorithms.

    Args:
        seed (int): The random seed to set.
        cudnn_deterministic (bool): If `True`, set CuDNN to use
            deterministic algorithms. Setting this to `True` can negatively
            impact performance, and might not be necessary for most cases.
            Defaults to `False`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_list(xs: OptionalList[T]) -> List[T]:
    if xs is None:
        return []
    if isinstance(xs, list):
        return xs
    return [xs]


def _to_metric_dict(d: OptionalDict[Metric]) -> Dict[str, Metric]:
    if d is None:
        return {}
    if isinstance(d, dict):
        return d
    xs = d
    if not isinstance(d, list):
        xs = [d]
    ret = {}
    counter: CounterType[str] = Counter()
    for x in xs:
        name = f"{x.__class__.__name__}.{x.pred_name}"
        if name not in counter:
            ret[name] = x
        else:
            cnt = counter[name]
            if cnt == 1:
                ret[f"{name}.1"] = ret[name]
                del ret[name]
            ret[f"{name}.{cnt + 1}"] = x
        counter.update([name])
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


class SavedTrainingState(NamedTuple):
    r"""The entire training state to save to or load from checkpoints."""
    model: Dict[str, torch.Tensor]
    optimizer: Dict[str, torch.Tensor]
    scheduler: Dict[str, Any]
    system_rng: Any
    numpy_rng: Any
    torch_rng: Any


class TrainingStatus(NamedTuple):
    pass


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
                 valid_data: Optional[DataBase] = None,
                 test_data: OptionalList[DataBase] = None,
                 batching_strategy: Optional[BatchingStrategy] = None,
                 device: Optional[torch.device] = None,
                 # Checkpoint
                 checkpoint_dir: Optional[str] = None,
                 max_to_keep: Optional[int] = None,
                 save_every: OptionalList[Condition] = None,
                 save_training_state: bool = True,  # whether to save optimizer, scheduler, RNG state
                 # Training
                 train_metrics: OptionalDict[Metric] = None,
                 optimizer: Optional[Instance[Optimizer]] = None,
                 lr_scheduler: Optional[Instance[LRScheduler]] = None,
                 max_epochs: Optional[int] = None,
                 num_backwards_per_update: int = 1,
                 grad_clip: Optional[float] = None,
                 # Validation
                 valid_metrics: OptionalDict[Metric] = None,
                 validate_every: OptionalList[Condition] = None,
                 early_stop_patience: Optional[int] = None,
                 plateau_condition: OptionalList[Condition] = None,
                 action_on_plateau: OptionalList[Action] = None,
                 validate_mode: str = 'eval',
                 # Testing
                 test_metrics: OptionalDict[Metric] = None,
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
        self.valid_data = valid_data
        self.test_data = _to_list(test_data)
        self.batching_strategy = batching_strategy

        # Device placement
        self._moved_model = False
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(torch.cuda.current_device())
            else:
                device = torch.device('cpu')
        self.device = device

        # Checkpoint management
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self._save_conditions = _to_list(save_every)
        self._save_training_state = save_training_state

        # Training loop
        self.train_metrics = _to_metric_dict(train_metrics)
        self.optimizer = _to_instance(
            Optimizer, optimizer, ["torch.optim", "texar.core"])
        self.lr_scheduler = _to_instance(
            LRScheduler, lr_scheduler,
            ["torch.optim.lr_scheduler", "texar.core"])
        self.max_epochs = max_epochs
        self.num_backwards_per_update = num_backwards_per_update
        self.grad_clip = grad_clip
        self._should_terminate = False

        # Validation
        self.valid_metrics = _to_metric_dict(valid_metrics)
        self._valid_conditions = _to_list(validate_every)
        self.valid_mode = validate_mode
        self.early_stop_patience = early_stop_patience
        self._plateau_conditions = _to_list(plateau_condition)
        self._actions_on_plateau = _to_list(action_on_plateau)

        # Testing
        self.test_metrics = _to_metric_dict(test_metrics)
        self._test_conditions = _to_list(test_every)
        self.test_mode = test_mode

        # Logging
        self._log_conditions = _to_list(log_every)
        self.log_format: str
        self.log_destination: List[Union[str, IO[str]]]
        self.eval_log_format: str

        for attr, default in self._defaults.items():
            value = locals()[attr]
            if value is None:
                value = default
            setattr(self, attr, value)

        # Initialize hooks.
        self._hooks: Dict[HookPoint,
                          Dict[Optional[Condition], List[Action]]] = {
            (event, point): defaultdict(list)
            for event in Event for point in (False, True)}

        # Register events & actions.
        for cond in self._valid_conditions:
            self.register_action(cond, self._validate)
        for cond in self._test_conditions:
            self.register_action(cond, self.test)
        for cond in self._log_conditions:
            self.register_action(cond, lambda: self._log(log_format))
        for event in (Event.Validation, Event.Testing):
            self._register_hook(
                (event, True), lambda: self._log(eval_log_format))
        for cond in self._save_conditions:
            self.register_action(cond, self._save)
        for cond in self._plateau_conditions:
            for action in self._actions_on_plateau:
                self.register_action(cond, action)

        # Detect event-action cycles.
        # TODO: Maybe don't do this.

        # Validate arguments.

    def _log(self, format_str: str):
        pass

    def register_action(self, cond: Condition, action: Action):
        for hook_point in cond._hooks:
            self._register_hook(hook_point, action, cond)

    def _register_hook(self, hook_point: HookPoint, action: Action,
                       cond: Optional[Condition] = None):
        self._hooks[hook_point][cond].append(action)

    @overload
    def on(self, event: Event, point: str = 'end') \
            -> Callable[[HookFn], HookFn]:
        ...

    @overload
    def on(self, event: Event, point: str, func: HookFn) -> None:
        ...

    def on(self, event, point='end', func=None):
        r"""Register a function as an event hook. For example:

        .. code-block:: python

            executor = Executor(...)

            @executor.on(Event.Epoch, 'end')
            def log_at_end_of_epoch(status):
                logging.info("Epoch %d done", status.epoch)

        Args:
            event: The event to hook on. Must be an enum value from
                :class:`~Event`.
            point (str): The point of event to hook on. Supported values are
                ``"begin"`` and ``"end"``.
            func (optional): The function to register. If not `None`, the
                function will be registered; if `None`, a decorate will be
                returned.

        :returns:
            - If :attr:`func` is `None`, this method will return a decorator to
              wrap around the function to register as hook.
            - If :attr:`func` is not `None`, this method will return `None`.
        """
        if not isinstance(event, Event) or point not in ('begin', 'end'):
            raise ValueError(f"Invalid hook point ({event}, {point})")
        end = (point == 'end')

        if func is not None:
            self._register_hook((event, end), func)
            return

        def wrapper(f):
            self._register_hook((event, end), f)
            return f

        return wrapper

    def _fire_event(self, event: Event, end: bool, *args, **kwargs):
        for cond, actions in self._hooks[(event, end)].items():
            # If condition is `None` (raw function hooks), action always
            # triggers.
            if cond is None or cond._hooks[(event, end)]():
                for action in actions:
                    action(*args, **kwargs)

    def _validate_step(self, batch: Batch):
        if self.valid_mode == 'predict':
            return_dict = self.model.predict(batch)
        else:
            return_dict = self.model(batch)
        return return_dict

    def _test_step(self, batch: Batch):
        if self.test_mode == 'predict':
            return_dict = self.model.predict(batch)
        else:
            return_dict = self.model(batch)
        return return_dict

    def _train_step(self, batch: Batch, epoch: int, iteration: int):
        return_dict = self.model(batch)
        try:
            loss = return_dict['loss']
        except KeyError:
            raise ValueError("Return dictionary from model does not "
                             "contain 'loss' entry")
        loss /= self.num_backwards_per_update
        loss.backward()
        if iteration % self.num_backwards_per_update == 0:
            self._fire_event(Event.ParameterUpdate, False)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self._fire_event(Event.ParameterUpdate, True)
        return return_dict

    def _save(self):
        # TODO: Save a meta file?

        if self._save_training_state and self.optimizer is not None:
            train_state = SavedTrainingState(
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
        if isinstance(checkpoint, SavedTrainingState):
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

    def _validate(self) -> None:
        self._fire_event(Event.Validation, False)

        # Initialize metrics.
        for metric in self.valid_metrics.values():
            metric.reset()

        iterator = DataIterator(self.valid_data, self.batching_strategy)

        if self.valid_mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        with torch.no_grad():
            for batch in iterator:
                self._fire_event(Event.ValidationIteration, False)
                return_dict = self._validate_step(batch)

                # Update metrics.
                for metric in self.valid_metrics.values():
                    metric.add(return_dict[metric.pred_name],
                               batch[metric.label_name])

                self._fire_event(Event.ValidationIteration, True)

        self._fire_event(Event.Validation, True)

    def terminate(self) -> None:
        r"""Terminate training. This method is intended to be called within
        event actions, an example use case would be to implement a custom
        early-stopping mechanism.

        It is guaranteed that no other events will be triggered once terminate
        is called. However, other events under the same hook is still called.
        """
        self._should_terminate = True

    def train(self):
        if self.max_epochs is None and self.early_stop_patience is None:
            warnings.warn(
                "Neither `max_epochs` nor early stopping is configured. Unless "
                "a custom event action calls `Executor.terminate()`, training "
                "will run indefinitely.")

        # Initialize optimizer.
        if self.optimizer is None:
            raise ValueError("Optimizer is not specified")
        self.optimizer.zero_grad()

        # Initialize dataset.
        if self.train_data is None:
            raise ValueError("No training dataset is specified")
        self.train_data.to(self.device)
        iterator = DataIterator(self.train_data, self.batching_strategy)
        if len(self._valid_conditions) > 0:
            if self.valid_data is None:
                raise ValueError("Validation will be performed, but no "
                                 "validation dataset is specified.")
            self.valid_data.to(self.device)

        # Move model to appropriate device
        if not self._moved_model:
            self.model.to(self.device)
        self.model.train()

        # Initialize metrics.
        for metric in self.train_metrics.values():
            metric.reset()

        # Main training
        self._should_terminate = False
        epoch = 0
        iteration = 0
        self._fire_event(Event.Training, False)
        while self.max_epochs is None or epoch < self.max_epochs:
            epoch += 1
            self._fire_event(Event.Epoch, False)

            for batch in iterator:
                self._fire_event(Event.Iteration, False)
                iteration += 1
                return_dict = self._train_step(batch, epoch, iteration)

                # Update metrics.
                for metric in self.train_metrics.values():
                    metric.add(return_dict[metric.pred_name],
                               batch[metric.label_name])

                self._fire_event(Event.Iteration, True, iteration)
            self._fire_event(Event.Epoch, True, epoch)
        self._fire_event(Event.Training, True)

    def test(self, dataset: OptionalList[DataBase] = None):
        if dataset is None and self.test_data is None:
            raise ValueError("No testing dataset is specified")
        datasets = self.test_data if dataset is None else _to_list(dataset)

        if self.test_mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        with torch.no_grad():
            for dataset in datasets:
                # Initialize metrics.
                for metric in self.test_metrics.values():
                    metric.reset()

                self._fire_event(Event.Testing, False)
                dataset.to(self.device)
                iterator = DataIterator(dataset, self.batching_strategy)

                for batch in iterator:
                    self._fire_event(Event.TestingIteration, False)
                    return_dict = self._test_step(batch)

                    # Update metrics.
                    for metric in self.test_.values():
                        metric.add(return_dict[metric.pred_name],
                                   batch[metric.label_name])

                    self._fire_event(Event.TestingIteration, True)
                self._fire_event(Event.Testing, True)
