import pickle
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional, Set, Union, overload

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer

from texar.torch.run import executor_utils as utils
from texar.torch.data.data.data_base import DataBase
from texar.torch.data.data.data_iterators import BatchingStrategy, DataIterator
from texar.torch.data.data.dataset_utils import Batch
from texar.torch.run.condition import Condition, Event, HookPoint
from texar.torch.run.executor_utils import Instance, OptionalDict, OptionalList
from texar.torch.run.metric import Metric

__all__ = [
    "make_deterministic",
    "Executor",
]

HookFn = Callable[['Executor'], None]


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


class Executor:
    _CHECKPOINT_METAINFO_FILE = "checkpoint.meta-info"
    _CHECKPOINT_EXTENSION = ".pt"
    _defaults: Dict[str, Any] = {
        "log_format": "{time} : Epoch {epoch} @ {iteration}it "
                      "({progress}%, {speed}), {{{metric:.3f}}}",
        "eval_log_format": "{time} : Epoch {epoch}, "
                           "{split} result = {{{metric:.3f}}}",
        "log_destination": [sys.stdout],
    }

    # TODO: Add loss as a special default metric? Otherwise would have to use
    #   RunningAverage with display steps

    _train_tracker: utils.ProgressTracker
    _eval_tracker: utils.ProgressTracker
    _hooks: Dict[HookPoint, Dict[Optional[Condition], List[HookFn]]]

    status: utils.TrainingStatus

    def __init__(self, model: nn.Module,
                 train_data: Optional[DataBase] = None,
                 *,
                 valid_data: Optional[DataBase] = None,
                 test_data: OptionalDict[DataBase] = None,
                 batching_strategy: Optional[BatchingStrategy] = None,
                 device: Optional[torch.device] = None,
                 # Checkpoint
                 checkpoint_dir: Optional[str] = None,
                 max_to_keep: Optional[int] = None,
                 save_every: OptionalList[Condition] = None,
                 save_training_state: bool = True,
                 # Training
                 train_metrics: OptionalDict[Metric] = None,
                 optimizer: Optional[Instance[Optimizer]] = None,
                 lr_scheduler: Optional[Instance[LRScheduler]] = None,
                 stop_training_on: OptionalList[Metric] = None,
                 num_iters_per_update: int = 1,
                 grad_clip: Optional[float] = None,
                 # Validation
                 valid_metrics: OptionalDict[Metric] = None,
                 validate_every: OptionalList[Condition] = None,
                 plateau_condition: OptionalList[Condition] = None,
                 action_on_plateau: OptionalList[HookFn] = None,
                 validate_mode: str = 'eval',
                 # Testing
                 test_metrics: OptionalDict[Metric] = None,
                 test_every: OptionalList[Condition] = None,
                 test_mode: str = 'predict',
                 # Logging
                 log_every: OptionalList[Condition] = None,
                 log_format: Optional[str] = None,
                 log_destination: OptionalList[Union[str, IO[str]]] = None,
                 print_configs_to_log: bool = True,
                 eval_log_format: Optional[str] = None,
                 show_progress_bar: bool = False,
                 # Tensorboard
                 # pylint: disable=unused-argument
                 tensorboard_log_dir: Optional[str] = None,
                 write_summary_every: OptionalList[Condition] = None
                 # pylint: enable=unused-argument
                 ):

        # TODO: Add support for Tensorboard.

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = utils.to_dict(test_data, default_name="test")
        self.batching_strategy = batching_strategy

        # Device placement
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(torch.cuda.current_device())
            else:
                device = torch.device('cpu')
        self.device = device
        self.model.to(device)

        # Logging
        self._log_conditions = utils.to_list(log_every)
        self._print_configs_to_log = print_configs_to_log
        self._show_progress_bar = show_progress_bar

        self.log_format: str = log_format or self._defaults["log_format"]
        self.eval_log_format: str = (
                eval_log_format or self._defaults["eval_log_format"])

        # Checkpoint management
        self.checkpoint_dir = (Path(checkpoint_dir)
                               if checkpoint_dir is not None else None)
        self.max_to_keep = max_to_keep
        self._save_conditions = utils.to_list(save_every)
        self._save_training_state = save_training_state

        self._directory_exists = False
        if self.checkpoint_dir is not None:
            if not self.checkpoint_dir.exists():
                self.checkpoint_dir.mkdir(parents=True)
            else:
                self._directory_exists = True

        # TODO: Close files somewhere? Maybe `atexit.register`?
        # Create logging files after checkpoint directory is created.
        self.log_destination: List[IO[str]] = [
            # We append to the logs to prevent accidentally overwriting previous
            # logs. To force overwrite, pass IO objects instead of file paths.
            open(dest, "a") if isinstance(dest, (str, Path)) else dest  # type: ignore
            for dest in utils.to_list(  # type: ignore
                log_destination or self._defaults["log_destination"])]

        # Training loop
        self.train_metrics = utils.to_metric_dict(train_metrics)
        self.optimizer = utils.to_instance(
            Optimizer, optimizer, ["torch.optim", "texar.core"],
            extra_kwargs={"params": self.model.parameters()})
        self.lr_scheduler = utils.to_instance(
            LRScheduler, lr_scheduler,
            ["torch.optim.lr_scheduler", "texar.core"],
            extra_kwargs={"optimizer": self.optimizer})
        self._stop_training_conditions = utils.to_list(stop_training_on)
        self.num_iters_per_update = num_iters_per_update
        self.grad_clip = grad_clip
        self._should_terminate = False

        # Validation
        self.valid_metrics = utils.to_metric_dict(valid_metrics)
        self._valid_conditions = utils.to_list(validate_every)
        self.valid_mode = validate_mode
        self._plateau_conditions = utils.to_list(plateau_condition)
        self._actions_on_plateau = utils.to_list(action_on_plateau)

        # Testing
        if (test_metrics is None and valid_metrics is not None and
                validate_mode == test_mode):
            self.test_metrics = self.valid_metrics
        else:
            self.test_metrics = utils.to_metric_dict(test_metrics)
        self._test_conditions = utils.to_list(test_every)
        self.test_mode = test_mode

        # Initialize hooks.
        self._hooks = {
            (event, point): defaultdict(list)
            for event in Event for point in (False, True)}

        self.status = {
            "epoch": 0,
            "iteration": 0,
            "split": "train",
            "metric": self.train_metrics,
            "eval_metric": self.valid_metrics,
        }

        # Register events & actions.
        train_logging_hook = self._create_logging_hook(
            self.log_format, eval_mode=False)
        for cond in self._log_conditions:
            self.register_action(cond, train_logging_hook)
        eval_logging_hook = self._create_logging_hook(
            self.eval_log_format, eval_mode=True)
        for event in (Event.Validation, Event.Testing):
            self._register_hook((event, True), eval_logging_hook)

        for cond in self._valid_conditions:
            self.register_action(cond, self._validate.__func__)  # type: ignore
        for cond in self._test_conditions:
            self.register_action(cond, self.test.__func__)  # type: ignore
        for cond in self._save_conditions:
            self.register_action(cond, self.save.__func__)  # type: ignore

        for cond in self._plateau_conditions:
            for action in self._actions_on_plateau:
                self.register_action(cond, action)

        for cond in self._stop_training_conditions:
            self.register_action(cond, self.terminate.__func__)  # type: ignore

        # Detect event-action cycles.
        # TODO: Maybe don't do this.

        # Validate arguments.

    def write_log(self, log_str: str, *, mode: str = "info"):
        if mode == "log":
            pass
        elif mode == "info":
            time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            log_str = utils.color(f"INFO {time_str} : ", "green") + log_str
        elif mode == "warning":
            time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            log_str = f"WARNING {time_str} : {log_str}"
            log_str = utils.color(log_str, "red")
        else:
            raise ValueError(f"Invalid logging mode {mode}")
        for dest in self.log_destination:
            # TODO: Strip color codes for non-terminals?
            dest.write(log_str)
            dest.write("\n")
            dest.flush()

    # TODO: Determine train/eval
    def _create_logging_hook(self, format_str: str, eval_mode: bool) -> HookFn:
        r"""Create a hook function given a logging format string.

        The format string follows the syntax of Python format strings. The
        status variables you can reference include:

        - ``epoch`` (int): The current epoch.
        - ``iteration`` (int): The current iteration.
        - ``progress`` (float): The epoch progress represented in percentage,
          i.e. a floating-point number between 0 and 100. It should be noted
          that progress may not be accurate, and may not be available if the
          data is loaded lazily.
        - ``speed`` (float): Average number of data examples processed per
          second. It should be noted that speed may not be accurate.
        - ``time``: The current date and time. Time format can be set using the
          "format spec" syntax of Python format strings, i.e.:
          ``{time:%H:%M:%S}`` prints time in the format ``08:26:03``. If time
          format is not specified, it is equivalent to
          ``{time:%Y-%m-%d %H:%M:%S}``, which corresponds to the format
          ``2018-07-06 08:26:03``. For more information on time formatting,
          please refer to documentation for Python built-in function
          :meth:`date.strftime`.
        - ``metric``: An aggregated representation of all metrics, in the
          format of ``<name1>: <value1>, <name2>: <value2>, ...``. The format
          spec for ``metric`` will be applied to all metrics whose value
          supports such format spec.
          For more fine grained control over metric formatting, use the
          following methods.
        - ``metric.<name>``: Value of the metric under the specified name
          ``<name>``.
        - ``<name>``: Value of the metric under the specified name ``<name>``.
          **Note:** The metric can only be looked-up if its name does not
          coincide with built-in status variables. For instance, a metric named
          "loss" can be looked-up by ``loss`` and ``metric.loss``, but a metric
          named "time" can only be looked-up by ``metric.time``.

        Args:
            format_str (str): The logging format string.
            eval_mode (bool): If `True`, returned hook function will be for
                logging during evaluation; if `False`, hook function will be for
                during training.

        Returns:
            A hook function to print logs given the format string.
        """
        format_var_regex = re.compile(r"{([a-zA-Z_0-9]+)(:([^{}]+))?}")
        format_vars: Dict[str, Optional[str]] = {
            match.group(1): match.group(3)
            for match in format_var_regex.finditer(format_str)
        }

        # Set default time format.
        if "time" in format_vars and format_vars["time"] is None:
            format_vars["time"] = "%Y-%m-%d %H:%M:%S"
            format_str = re.sub(r"{time(:([^{}]+))?}", "{time}", format_str)

        # Set metric print formats for aggregated format variable.
        metrics = (self.status["eval_metric"]
                   if eval_mode else self.status["metric"])
        if "metric" in format_vars and format_vars["metric"] is not None:
            # Check which metrics can be printed with specified format.
            fmt_metrics: Set[str] = set()
            metric_format = format_vars["metric"]
            for name, metric in metrics.items():
                try:
                    metric_format.format(metric.value())
                    fmt_metrics.add(name)
                except ValueError:
                    pass
            metric_format_parts = []
            for name in metrics:
                metric_format_parts.append(
                    f"{name}: {{{name}"
                    f"{':' + metric_format if name in fmt_metrics else ''}}}")
            metric_format_str = ', '.join(metric_format_parts)
            format_vars["metric"] = metric_format_str
            format_str = re.sub(r"{metric(:([^{}]+))?}", "{metric}", format_str)

        # Gather metrics represented as "name" or "metric.name".
        metrics_to_print: Set[str] = set()
        for name in format_vars:
            if name in self.status or name in ["time", "progress", "speed"]:
                # built-in name
                pass
            elif name in metrics:
                metrics_to_print.add(name)
            else:
                raise ValueError(
                    f"Invalid status variable name '{name}' in format string")

        format_str_wo_progress = re.sub(
            r"{progress(:([^{}]+))?}", "{progress}", format_str)

        def log_fn(self):
            metrics = (self.status["eval_metric"]
                       if eval_mode else self.status["metric"])
            format_args = self.status.copy()
            cur_format_str = format_str
            if "time" in format_vars:
                format_args["time"] = time.strftime(format_vars["time"])
            if "metric" in format_vars:
                metric_vals = {name: metric.value()
                               for name, metric in metrics.items()}
                metric_str = format_vars["metric"].format(**metric_vals)
                format_args["metric"] = metric_str
            if "speed" in format_vars:
                format_args["speed"] = self._train_tracker.speed()
            if "progress" in format_vars:
                progress = self._train_tracker.progress()
                if progress is not None:
                    if format_vars["progress"] is None:
                        progress = round(progress, 1)
                    format_args["progress"] = progress
                else:
                    cur_format_str = format_str_wo_progress
                    format_args["progress"] = "unknown"
            format_args.update({name: metrics[name].value()
                                for name in metrics_to_print})
            log_str = cur_format_str.format(**format_args)

            self.write_log(log_str, mode="log")

        return log_fn

    def register_action(self, cond: Condition, action: HookFn):
        for hook_point in cond.hooks:
            self._register_hook(hook_point, action, cond)
        return self

    def _register_hook(self, hook_point: HookPoint, action: HookFn,
                       cond: Optional[Condition] = None):
        try:
            self._hooks[hook_point][cond].append(action)
        except KeyError:
            raise ValueError(
                f"Specified hook point {hook_point} is invalid") from None

    # pylint: disable=unused-argument,no-self-use,function-redefined

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
            def log_at_end_of_epoch(executor):
                logging.info("Epoch %d done", executor.status.epoch)

        The hook function takes exactly one argument: the executor instance
        itself.

        Args:
            event: The event to hook on. Must be an enum value from
                :class:`~Event`.
            point (str): The point of event to hook on. Supported values are
                ``"begin"`` and ``"end"``.
            func (optional): The function to register. If not `None`, the
                function will be registered; if `None`, a decorator will be
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

    # pylint: enable=unused-argument,no-self-use,function-redefined

    def _fire_event(self, event: Event, end: bool):
        for cond, actions in self._hooks[(event, end)].items():
            # If condition is `None` (raw function hooks), action always
            # triggers.
            if cond is None or cond.hooks[(event, end)](self):
                for action in actions:
                    action(self)
        if self._should_terminate:
            self._should_terminate = False
            raise utils.TerminateExecution

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

    def _train_step(self, batch: Batch):
        return_dict = self.model(batch)
        try:
            loss = return_dict['loss']
        except KeyError:
            raise ValueError("Return dictionary from model does not "
                             "contain 'loss' entry")
        loss /= self.num_iters_per_update
        loss.backward()
        if self.status["iteration"] % self.num_iters_per_update == 0:
            self._fire_event(Event.ParameterUpdate, False)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # type: ignore
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # type: ignore
            self.optimizer.zero_grad()  # type: ignore
            self._fire_event(Event.ParameterUpdate, True)
        return return_dict

    def save(self):
        # Load the checkpoint meta-info file.
        meta_path = self.checkpoint_dir / self._CHECKPOINT_METAINFO_FILE
        if meta_path.exists():
            try:
                with meta_path.open("rb") as f:
                    meta_dict = pickle.load(f)
            except (EOFError, IOError):
                meta_dict = {}
        else:
            meta_path.touch()
            meta_dict = {}

        # Remove earliest checkpoints if exceeds `max_to_keep`.
        if self.max_to_keep is not None:
            while len(meta_dict) >= self.max_to_keep:
                checkpoint_name = min(
                    meta_dict, key=lambda name: meta_dict[name]["timestamp"])
                (self.checkpoint_dir / checkpoint_name).unlink()
                del meta_dict[checkpoint_name]
                self.write_log(f"Previous checkpoint {checkpoint_name} removed "
                               f"due to `max_to_keep`(={self.max_to_keep}) "
                               f"limit", mode="info")

        timestamp = time.time()
        checkpoint_name = str(timestamp) + self._CHECKPOINT_EXTENSION
        path = self.checkpoint_dir / checkpoint_name
        if path.exists():
            idx = 0
            while True:
                checkpoint_name = (
                        str(timestamp) + f".{idx}" + self._CHECKPOINT_EXTENSION)
                path = self.checkpoint_dir / checkpoint_name
                if not path.exists():
                    break
        if self._save_training_state and self.optimizer is not None:
            train_state = utils.SavedTrainingState(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=(self.lr_scheduler.state_dict()
                           if self.lr_scheduler is not None else None),
                system_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                torch_rng=torch.random.get_rng_state(),
            )
            torch.save(train_state, str(path))
        else:
            torch.save(self.model.state_dict(), str(path))
        meta_dict[checkpoint_name] = {
            "status": self.status,
            "timestamp": timestamp,
        }
        with meta_path.open("wb") as f:
            pickle.dump(meta_dict, f)

        self.write_log(f"Current checkpoint saved to {path}", mode="info")

    def load(self, path: Optional[str] = None,
             load_training_state: bool = True,
             allow_failure: bool = False) -> Optional[Path]:
        r"""Load a previous model checkpoint from file.

        Args:
            path (str, optional): Path to a specific checkpoint or a checkpoint
                directory. If a directory is specified, the most recent
                checkpoint in the directory is loaded. If `None`,
                :attr:`checkpoint_dir` will be used.
            load_training_state (bool): If `True`, will load entire training
                state from checkpoint (if the checkpoint contains training
                state). Otherwise, just load model weights. Defaults to `True`.
            allow_failure (bool): If `True`, no exceptions will be raised if
                no checkpoints were found. Defaults to `False`. Note that
                exceptions are still raised if the provided :attr:`path` points
                to a file, or the selected checkpoint is corrupted.

        Returns:
            Path of the loaded checkpoint, or `None` if load failed.
        """
        if path is not None:
            ckpt_path = Path(path)
        elif self.checkpoint_dir is not None:
            ckpt_path = self.checkpoint_dir
        else:
            raise ValueError(
                "`path` must be specified when `checkpoint_dir` is `None`")
        if ckpt_path.is_dir():
            try:
                meta_path = ckpt_path / self._CHECKPOINT_METAINFO_FILE
                if meta_path.exists():
                    with meta_path.open("rb") as f:
                        meta_dict = pickle.load(f)
                    # TODO: Metric values may be stale. Also include timestamp?
                    best_metric, best_m_time, best_ckpt_name = max(
                        (utils.MetricList(info["status"]["eval_metric"]),
                         info["timestamp"], name)
                        for name, info in meta_dict.items())
                    status = meta_dict[best_ckpt_name]["status"]
                    ckpt_path = self.checkpoint_dir / best_ckpt_name
                    metric_vals = [(name, best_metric.values[name])
                                   for name in best_metric.metrics]
                    metric_str = ", ".join(
                        f"{name}: " + (
                            f"{value:.3f}" if isinstance(value, float)
                            else f"{value}")
                        for name, value in metric_vals)
                    time_str = datetime.fromtimestamp(best_m_time).strftime(
                        "%Y-%m-%d %H:%M:%S")
                    load_info_str = (f"saved at: {time_str}, "
                                     f"meta-info: epoch={status['epoch']}, "
                                     f"iteration={status['iteration']}, "
                                     f"metrics={{{metric_str}}}")
                else:
                    m_time, ckpt_path = max(
                        (name.stat().st_mtime, name)
                        for name in ckpt_path.iterdir()
                        if name.suffix == self._CHECKPOINT_EXTENSION)
                    time_str = datetime.fromtimestamp(m_time).strftime(
                        "%Y-%m-%d %H:%M:%S")
                    load_info_str = f"saved at: {time_str}"
                    self.write_log(
                        "Checkpoint meta-info not found. Will load the most "
                        "recent checkpoint in directory", mode="warning")
            except (EOFError, IOError, ValueError):
                # EOFError, IOError: pickle.load
                # ValueError: max() arg is an empty sequence
                if allow_failure:
                    return None
                raise
        else:
            load_info_str = ""

        checkpoint = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(checkpoint, utils.SavedTrainingState):
            self.model.load_state_dict(checkpoint.model)
            if load_training_state:
                # TODO: Also somehow save/load data iterator state?
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint.optimizer)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint.scheduler)
                random.setstate(checkpoint.system_rng)
                np.random.set_state(checkpoint.numpy_rng)
                torch.random.set_rng_state(checkpoint.torch_rng.cpu())
        else:
            self.model.load_state_dict(checkpoint)

        if load_info_str != "":
            self.write_log(f"Checkpoint ({load_info_str}) "
                           f"loaded from {ckpt_path}.", mode="info")
        else:
            self.write_log(f"Checkpoint loaded from {ckpt_path}.", mode="info")
        return ckpt_path

    def _validate(self) -> None:
        if self.valid_data is None:
            raise ValueError("Validation data not specified.")

        self._fire_event(Event.Validation, False)

        # Initialize metrics.
        for metric in self.valid_metrics.values():
            metric.reset()

        if self.valid_mode == "eval":
            iterator = DataIterator(self.valid_data, self.batching_strategy)
        else:
            iterator = DataIterator(self.valid_data)

        try:
            data_size: Optional[int] = len(self.valid_data)
        except TypeError:
            data_size = None
        self._eval_tracker = utils.ProgressTracker(data_size)

        if self.valid_mode == "train":
            self.model.train()
        else:
            self.model.eval()

        prev_split = self.status["split"]
        self.status["split"] = "valid"

        with torch.no_grad():
            for batch in iterator:
                self._fire_event(Event.ValidationIteration, False)
                return_dict = self._validate_step(batch)

                # Update metrics.
                utils.update_metrics(return_dict, batch, self.valid_metrics)

                self._fire_event(Event.ValidationIteration, True)

        self._fire_event(Event.Validation, True)

        self.status["split"] = prev_split

    def terminate(self) -> None:
        r"""Terminate training. This method is intended to be called within
        event actions, an example use case would be to implement a custom
        early-stopping mechanism.

        It is guaranteed that no other events will be triggered once terminate
        is called. However, other events under the same hook is still called.
        """
        self._should_terminate = True

    def train(self):
        # TODO: Print the model architecture somewhere?

        if self._directory_exists:
            self.write_log(
                f"Specified checkpoint directory '{self.checkpoint_dir}' "
                f"exists, previous checkpoints might be erased", mode="warning")

        if len(self._stop_training_conditions) == 0:
            self.write_log(
                "`stop_training_on` is not configured. Unless an event action "
                "calls `executor.terminate()`, training will run indefinitely.",
                mode='warning')

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

        # Move model to appropriate device.
        self.model.train()
        self.status["split"] = "train"

        epoch = 0
        iteration = 0
        self._fire_event(Event.Training, False)
        self.write_log("Training started", mode="info")

        # Main training loop.
        try:
            while True:
                try:
                    data_size: Optional[int] = len(self.train_data)
                except TypeError:
                    data_size = None
                self._train_tracker = utils.ProgressTracker(data_size)

                epoch += 1
                self.status["epoch"] = epoch
                # Initialize metrics.
                for metric in self.train_metrics.values():
                    metric.reset()

                self._fire_event(Event.Epoch, False)

                for batch in iterator:
                    self._fire_event(Event.Iteration, False)
                    iteration += 1
                    self.status["iteration"] = iteration

                    return_dict = self._train_step(batch)

                    self._train_tracker.add(len(batch))
                    utils.update_metrics(return_dict, batch, self.train_metrics)

                    self._fire_event(Event.Iteration, True)
                self._fire_event(Event.Epoch, True)
        except utils.TerminateExecution:
            self.write_log("Training terminated", mode='info')
        self._fire_event(Event.Training, True)

    def test(self, dataset: OptionalDict[DataBase] = None):
        if dataset is None and self.test_data is None:
            raise ValueError("No testing dataset is specified")
        if len(self.test_metrics) == 0:
            raise ValueError(
                "No testing metric is specified. Validation metrics are not "
                "used due to different modes for validation and test")
        datasets = (self.test_data if dataset is None
                    else utils.to_dict(dataset, default_name="test"))

        if self.test_mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        with torch.no_grad():
            for name, data in datasets.items():
                try:
                    self.status["split"] = name

                    # Initialize metrics.
                    for metric in self.test_metrics.values():
                        metric.reset()

                    self._fire_event(Event.Testing, False)
                    data.to(self.device)
                    if self.test_mode == "eval":
                        iterator = DataIterator(data, self.batching_strategy)
                    else:
                        iterator = DataIterator(data)
                    try:
                        data_size: Optional[int] = len(data)
                    except TypeError:
                        data_size = None
                    self._eval_tracker = utils.ProgressTracker(data_size)

                    for batch in iterator:
                        self._fire_event(Event.TestingIteration, False)
                        return_dict = self._test_step(batch)

                        self._eval_tracker.add(len(batch))
                        utils.update_metrics(
                            return_dict, batch, self.test_metrics)

                        self._fire_event(Event.TestingIteration, True)
                except utils.TerminateExecution:
                    self.write_log(
                        "Testing terminated. This is likely unintended. Please "
                        "check your custom actions.", mode='warning')
                self._fire_event(Event.Testing, True)
