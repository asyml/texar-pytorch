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
Utility functions related to variables.
"""

from typing import Any, List, Tuple, Union

__all__ = [
    "add_variable",
    "collect_trainable_variables"
]


def add_variable(
        variable: Union[List[Any], Tuple[Any]],
        var_list: List[Any]):
    r"""Adds variable to a given list.

    Args:
        variable: A (list of) variable(s).
        var_list (list): The list where the :attr:`variable` are added to.
    """
    if isinstance(variable, (list, tuple)):
        for var in variable:
            add_variable(var, var_list)
    else:
        var_list.append(variable)


def collect_trainable_variables(
        modules: Union[Any, List[Any]]
):
    r"""Collects all trainable variables of modules.

    Trainable variables included in multiple modules occur only once in the
    returned list.

    Args:
        modules: A (list of) instance of the subclasses of
            :class:`~texar.torch.modules.ModuleBase`.

    Returns:
        A list of trainable variables in the modules.
    """
    if not isinstance(modules, (list, tuple)):
        modules = [modules]

    var_list: List[Any] = []
    for mod in modules:
        add_variable(mod.trainable_variables, var_list)

    return var_list
