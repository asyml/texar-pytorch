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
Model utilities.
"""

from typing import Callable


__all__ = [
    "warmup_lr_lambda",
]


def warmup_lr_lambda(total_steps: int, warmup_steps: int = 0,
                     min_lr_ratio: float = 0.0) -> Callable[[int], float]:
    r"""Create a learning rate schedule with a linear warm-up stage and linear
    decay.

    Args:
        total_steps (int): The total number of training steps.
        warmup_steps (int): The number of steps in the warm-up stage.
        min_lr_ratio (float): The LR at the end of training, represented as the
            percentage of the base LR.

    :return: A lambda function compatible with
        `torch.optim.lr_scheduler.LambdaLR`.
    """

    def polynomial_lr(decay_steps: int, step: int) -> float:
        return (1.0 - min_lr_ratio) * (1 - step / decay_steps) + min_lr_ratio

    if warmup_steps == 0:
        return lambda step: polynomial_lr(total_steps, step + 1)

    def lambda_lr(step: int) -> float:
        step += 1
        if step <= warmup_steps:
            return step / warmup_steps
        return polynomial_lr(total_steps - warmup_steps, step - warmup_steps)

    return lambda_lr
