# Copyright 2020 The Texar Authors. All Rights Reserved.
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

from typing import Optional

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from texar.torch.run.executor_utils import Instance

import adaptdl.torch


class AdaptiveDataParallel(adaptdl.torch.AdaptiveDataParallel):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: Optimizer,
                 lr_scheduler: Optional[Instance[LRScheduler]] = None,
                 **kwargs):
        super().__init__(model, optimizer, lr_scheduler, **kwargs)

        # Add missing members from model
        missing = {k: model.__dict__[k]
                   for k in set(model.__dict__) - set(self.__dict__)}
        self.__dict__.update(missing)
