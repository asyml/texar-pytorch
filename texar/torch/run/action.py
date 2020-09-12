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
Actions for the Executor module.
"""

from abc import ABC, abstractmethod

from texar.torch.run.executor import Executor


__all__ = [
    "reset_params",
    "scale_lr",
    "early_stop",
]


class Action(ABC):
    @abstractmethod
    def __call__(self, executor: Executor) -> None:
        raise NotImplementedError


class reset_params(Action):
    def __init__(self, training_state: bool = True):
        self.load_training_state = training_state

    def __call__(self, executor: Executor):
        # TODO: Only optimizer?
        executor.load(load_training_state=self.load_training_state)


class scale_lr(Action):
    def __init__(self, scale: float):
        # TODO: Change to accept LRScheduler.
        self.scale = scale

    def __call__(self, executor: Executor):
        new_lr = []
        assert executor.optimizer is not None
        for group in executor.optimizer.param_groups:  # type: ignore
            lr = group['lr'] * self.scale
            group['lr'] = lr
            new_lr.append(lr)

        lr_str = ", ".join(f"{lr:.4e}" for lr in new_lr)
        if len(new_lr) > 1:
            lr_str = f"[{lr_str}]"
        executor.write_log(f"Learning rate scaled by {self.scale}. "
                           f"New LR: {lr_str}")


class early_stop(Action):
    def __init__(self, patience: int):
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError("`patience` must be a positive integer")
        self.patience = patience
        self.count = 0

    def __call__(self, executor: Executor):
        self.count += 1
        if self.count >= self.patience:
            executor.terminate()
            executor.write_log(f"Early stopping patience exhausted, "
                               f"terminating training...")
        else:
            executor.write_log(f"Early stopping patience decrease to "
                               f"{self.patience - self.count}")
