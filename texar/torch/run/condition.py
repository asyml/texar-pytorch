import functools
import types
from enum import Enum, auto
from time import time as time_now
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "Event",
    "HookPoint",
    "Condition",
    "epoch",
    "iteration",
    "validation",
    "consecutive",
    "time",
]


class StrEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Event(StrEnum):
    Iteration = auto()
    Epoch = auto()
    Training = auto()
    Validation = auto()
    Testing = auto()


HookPoint = Tuple[Event, str]


class Condition:
    _hooks: Dict[HookPoint, Any]

    def __init__(self):
        self._hooks = {}
        for name in self.__dict__:
            if not name.startswith("check_"):
                continue
            parts = name.split("_")
            if len(parts) != 3:
                continue
            event = Event(parts[1])
            point = parts[2]
            if point not in ['begin', 'end']:
                raise ValueError(
                    "Final part of hook name must be 'begin' or 'end'")
            self._hooks[(event, point)] = self.__dict__[name]


class epoch(Condition):
    r"""Triggers when the specified number of epochs has ended.

    Args:
        num_epochs (int): The number of epochs to wait before triggering the
            event. In other words, the event is triggered every
            :attr:`num_epochs` epochs.
    """

    def __init__(self, num_epochs: int = 1):
        super().__init__()
        self.num_epochs = num_epochs
        self.count = 0

    def check_epoch_end(self) -> bool:
        self.count += 1
        if self.count == self.num_epochs:
            self.count = 0
            return True
        return False


class iteration(Condition):
    r"""Triggers when the specified number of iterations had ended.

    Args:
        num_iters (int): The number of iterations to wait before triggering the
            event. In other words, the event is triggered every
            :attr:`num_iters` iterations.
    """

    def __init__(self, num_iters: int = 1):
        super().__init__()
        self.num_iters = num_iters
        self.count = 0

    def check_iteration_end(self) -> bool:
        self.count += 1
        if self.count == self.num_iters:
            self.count = 0
            return True
        return False


class validation(Condition):
    r"""Triggers when validation ends, and optionally checks if validation
    results improve or worsen.

    Args:
        better (bool, optional): If `True`, this event only triggers when
            validation results improve; if `False`, only triggers when results
            worsen. Defaults to `None`, in which case the event triggers
            regardless of results.
    """

    def __init__(self, better: Optional[bool] = None):
        super().__init__()
        self.better = better

    def check_validation_end(self, metrics) -> bool:
        pass


class consecutive(Condition):
    r"""Triggers when the specified condition passes checks for several times
    consecutively.

    For example: :python:`consecutive(validation(better=False), times=3)` would
    trigger if validation results do not improve for 3 times in a row.

    .. warning::

        This method might not work as you would expect.
        # TODO: Think about whether this is a good design.

    Args:
        cond: The base condition to check.
        times (int): The number of times the base condition should pass checks
            consecutively.
        clear_after_trigger (bool): Whether the counter should be cleared after
            the event is triggered. If :attr:`clear_after_trigger` is set to
            `False`, once this event is triggered, it will trigger every time
            :attr:`cond` is triggered, until :attr:`cond` fails to trigger at
            some point. Defalts to `True`.
    """

    def __init__(self, cond: Condition, times: int,
                 clear_after_trigger: bool = True):
        super().__init__()
        self.cond = cond
        self.times = times
        self.count = 0
        self.clear_after_trigger = clear_after_trigger

        for hook_point, method in self.cond._hooks.items():
            self._hooks[hook_point] = self._create_check_method(method)

    def _create_check_method(self, method):
        @functools.wraps(method)
        def check_fn(self, *args, **kwargs) -> bool:
            if method(*args, **kwargs):
                self.count += 1
                if self.count >= self.times:
                    if self.clear_after_trigger:
                        self.count = 0
                    return True
            else:
                self.count = 0
            return False

        return types.MethodType(check_fn, self)


class time(Condition):
    def __init__(self, *, hours: Optional[float] = None,
                 minutes: Optional[float] = None,
                 seconds: Optional[float] = None,
                 only_training: bool = True):
        super().__init__()
        self.seconds = 0.0
        if hours is not None:
            self.seconds += hours * 3600.0
        if minutes is not None:
            self.seconds += minutes * 60.0
        if seconds is not None:
            self.seconds += seconds
        self.only_training = only_training
        self.start_time: Optional[float] = None
        self.accumulated_time = 0.0

    def _should_trigger(self) -> bool:
        total_time = self.accumulated_time
        if self.start_time is not None:
            cur_time = time_now()
            total_time += cur_time - self.start_time
        else:
            cur_time = None
        self.start_time = cur_time
        if total_time >= self.seconds:
            self.accumulated_time = 0.0
            return True
        else:
            self.accumulated_time = total_time
            return False

    def check_training_begin(self) -> bool:
        self.start_time = time_now()
        return False

    def check_training_end(self) -> bool:
        return self._should_trigger()

    def check_validation_begin(self) -> bool:
        if self.only_training:
            self.accumulated_time += time_now() - self.start_time
            self.start_time = None
        return self._should_trigger()

    def check_validation_end(self) -> bool:
        if self.only_training:
            self.start_time = time_now()
            return False
        else:
            return self._should_trigger()

    def check_testing_begin(self) -> bool:
        if self.only_training:
            self.accumulated_time += time_now() - self.start_time
            self.start_time = None
        return self._should_trigger()

    def check_testing_end(self) -> bool:
        if self.only_training:
            self.start_time = time_now()
            return False
        else:
            return self._should_trigger()

    def check_iteration_end(self) -> bool:
        return self._should_trigger()
