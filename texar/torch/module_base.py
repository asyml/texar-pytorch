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
from abc import ABC
from typing import Any, Dict, List, Optional, Union

from torch import nn

from texar.torch.hyperparams import HParams

__all__ = [
    'ModuleBase',
]


class ModuleBase(nn.Module, ABC):
    r"""Base class inherited by modules that are configurable through
    hyperparameters.

    This is a subclass of :torch_nn:`Module`.

    A Texar module inheriting :class:`~texar.torch.ModuleBase` is
    **configurable through hyperparameters**. That is, each module defines
    allowed hyperparameters and default values. Hyperparameters not
    specified by users will take default values.

    Args:
        hparams (dict, optional): Hyperparameters of the module. See
            :meth:`default_hparams` for the structure and default values.
    """

    def __init__(self, hparams: Optional[Union[HParams,
                                               Dict[str, Any]]] = None):
        super().__init__()
        if not hasattr(self, '_hparams'):
            self._hparams = HParams(hparams, self.default_hparams())
        else:
            # Probably already parsed by subclasses. We rely on subclass
            # implementations to get this right.
            # As a sanity check, we require `hparams` to be `None` in this case.
            if hparams is not None:
                raise ValueError(
                    "`self._hparams` is already assigned, but `hparams` "
                    "argument is not None.")

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
            'name': 'module'
        }

    @property
    def trainable_variables(self) -> List[nn.Parameter]:
        r"""The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.

        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.
        """
        return [x for x in self.parameters() if x.requires_grad]

    @property
    def hparams(self) -> HParams:
        r"""An :class:`~texar.torch.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output tensor(s),
        usually it is equal to the last dimension value of the output
        tensor size.
        """
        raise NotImplementedError
