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
Modules of Texar library.
"""

# pylint: disable=wildcard-import

from texar.torch.version import VERSION as __version__

from texar.torch import core
from texar.torch import data
from texar.torch import evals
from texar.torch import losses
from texar.torch import modules
from texar.torch import run
from texar.torch import utils
from texar.torch.hyperparams import *
from texar.torch.module_base import *
