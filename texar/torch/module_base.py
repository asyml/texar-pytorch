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
from typing import Any, Dict, List, Optional, Union, NamedTuple

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

        Both parameters of this module and those of all submodules are included.
        """
        # TODO: The list returned may contain duplicate parameters (e.g. output
        #   layer shares parameters with embeddings). For most usages, it's not
        #   necessary to ensure uniqueness.
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

        def _get_indent(s: str) -> int:
            return len(s) - len(s.lstrip(' '))

        class ModuleRepr(NamedTuple):
            indent: int
            repr_str: str
            names: List[str]

        def _convert_repr(module: ModuleRepr) -> List[str]:
            prefix = (f"{' ' * module.indent}(id" +
                      (f"s {module.names[0]}-{module.names[-1]}"
                       if len(module.names) > 1 else f" {module.names[0]}") +
                      "): ")
            lines = module.repr_str.split('\n')
            lines[0] = prefix + lines[0]
            return lines

        repr_str = super().__repr__().split('\n')
        nested = True
        while nested:
            nested = False
            output_str = []
            prev_module: Optional[ModuleRepr] = None
            for idx, line in enumerate(repr_str):
                line = repr_str[idx]
                indent = _get_indent(line)
                if prev_module is not None and prev_module.indent > indent:
                    output_str.extend(_convert_repr(prev_module))
                    prev_module = None
                name = line[(indent + 1):line.find(')')]
                if line[indent] != '(' or not name.isnumeric():
                    if prev_module is None:
                        output_str.append(line)
                    continue

                end_idx = next(
                    end_idx for end_idx in range(idx + 1, len(repr_str))
                    if _get_indent(repr_str[end_idx]) <= indent)
                end_indent = _get_indent(repr_str[end_idx])
                if end_indent < indent or repr_str[end_idx][end_indent] != ')':
                    # not a module; a parameter in ParameterList
                    end_idx -= 1
                    indent -= 2  # parameters are somehow indented further
                module_repr = '\n'.join(
                    [line[(indent + len(name) + 4):]] +  # "(): "
                    repr_str[(idx + 1):(end_idx + 1)])
                if prev_module is None:
                    prev_module = ModuleRepr(indent, module_repr, [name])
                elif prev_module.indent < indent:
                    nested = True
                elif prev_module.repr_str == module_repr:
                    prev_module.names.append(name)
                else:
                    output_str.extend(_convert_repr(prev_module))
                    prev_module = ModuleRepr(indent, module_repr, [name])
            repr_str = output_str
        return '\n'.join(repr_str)
