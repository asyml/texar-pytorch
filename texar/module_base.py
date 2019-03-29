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
Base class for modules.
"""
from typing import Optional, Dict, Any, List

from torch import nn

from texar.hyperparams import HParams

__all__ = [
    "ModuleBase"
]


class ModuleBase(nn.Module):
    r"""Base class inherited by modules that are configurable through
    hyperparameters.

    This is a subclass of :torch_main:`torch.nn.Module <nn/modules/module>`.

    A Texar module inheriting :class:`~texar.ModuleBase` is
    **configurable through hyperparameters**. That is, each module defines
    allowed hyperparameters and default values. Hyperparameters not
    specified by users will take default values.

    Args:
        hparams (dict, optional): Hyperparameters of the module. See
            :meth:`default_hparams` for the structure and default values.
    """

    def __init__(self, hparams: Optional[HParams] = None):
        super(ModuleBase, self).__init__()
        self._hparams = HParams(hparams, self.default_hparams())

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a `dict` of hyperparameters of the module with default
        values. Used to replace the missing values of input `hparams`
        during module construction.

        .. code-block:: python

            {
                "name": "module"
            }
        """
        return {
            "name": "module"
        }

    def forward(self, *input):  # pylint: disable=redefined-builtin
        raise NotImplementedError

    @property
    def trainable_variables(self) -> List[nn.Parameter]:
        r"""The list of trainable variables (parameters) of the module.

        Both parameters of this module and those of all submodules are included.
        """
        return [x for x in self.parameters(recurse=True)]  # pylint: disable=unexpected-keyword-arg

    @property
    def hparams(self) -> HParams:
        r"""An :class:`~texar.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams
