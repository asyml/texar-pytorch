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


def _add_indent(s_, n_spaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(n_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


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

    def __repr__(self):
        r"""Create a compressed representation by combining identical modules in
        `nn.ModuleList`s and `nn.ParameterList`s.
        """

        def _convert_id(keys: List[str]) -> List[str]:
            start = end = None
            for key in keys:
                if key.isnumeric() and end == int(key) - 1:
                    end = int(key)
                else:
                    if start is not None:
                        if start == end:
                            yield f"id {start}"
                        else:
                            yield f"ids {start}-{end}"
                    if key.isnumeric():
                        start = end = int(key)
                    else:
                        start = end = None
                        yield key
            if start is not None:
                if start == end:
                    yield f"id {start}"
                else:
                    yield f"ids {start}-{end}"

        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        prev_mod_str = None
        keys = []
        for key, module in self._modules.items():
            if isinstance(module, ModuleBase):
                mod_str = repr(module)
            else:
                mod_str = ModuleBase.__repr__(module)
            mod_str = _add_indent(mod_str, 2)
            if prev_mod_str is None or prev_mod_str != mod_str:
                if prev_mod_str is not None:
                    for name in _convert_id(keys):
                        child_lines.append(f"({name}): {prev_mod_str}")
                prev_mod_str = mod_str
                keys = [key]
            else:
                keys.append(key)
        if len(keys) > 0:
            for name in _convert_id(keys):
                child_lines.append(f"({name}): {prev_mod_str}")
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
