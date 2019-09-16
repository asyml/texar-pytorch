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
Miscellaneous Utility functions.
"""

import collections
import copy
import inspect
from functools import lru_cache
from pydoc import locate
from typing import (
    Any, Callable, Collection, Dict, List, MutableMapping, Optional, Sequence,
    Tuple, Type, TypeVar, Union, cast, no_type_check, overload)

import funcsigs
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from texar.torch.hyperparams import HParams
from texar.torch.utils.dtypes import _maybe_list_to_array
from texar.torch.utils.types import MaybeSeq, MaybeTuple

MAX_SEQ_LENGTH = np.iinfo(np.int32).max

__all__ = [
    'no_map',
    'map_structure',
    'map_structure_zip',
    'get_first_in_structure',
    'sequence_mask',
    'get_args',
    'get_default_arg_values',
    'check_or_get_class',
    'get_class',
    'check_or_get_instance',
    'get_instance',
    'check_or_get_instance_with_redundant_kwargs',
    'get_instance_with_redundant_kwargs',
    'get_function',
    'call_function_with_redundant_kwargs',
    'get_instance_kwargs',
    'dict_patch',
    'dict_lookup',
    'dict_fetch',
    'dict_pop',
    'flatten_dict',
    'strip_token',
    'strip_eos',
    'strip_bos',
    'strip_special_tokens',
    'str_join',
    'default_str',
    'uniquify_str',
    'ceildiv',
    'sum_tensors',
]

T = TypeVar('T')  # type argument
R = TypeVar('R')  # return type
K = TypeVar('K')  # key type
V = TypeVar('V')  # value type
Kwargs = Dict[str, Any]
AnyDict = MutableMapping[str, Any]
ParamDict = Union[HParams, AnyDict]

Type_size_lambda_map = {
    nn.Linear: lambda x: x.out_features,
    nn.Bilinear: lambda x: x.out_features,
    _ConvNd: lambda x: x.out_channels * len(x.kernel_size),
    nn.Embedding: lambda x: x.embedding_dim,
    nn.EmbeddingBag: lambda x: x.embedding_dim,
    nn.RNNCellBase: lambda x: x.hidden_size,
}

Type_size_keeper = [
    nn.ELU, nn.Hardshrink, nn.Hardtanh, nn.LeakyReLU, nn.LogSigmoid,
    nn.PReLU, nn.ReLU, nn.RReLU, nn.SELU, nn.CELU, nn.Sigmoid, nn.Softplus,
    nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Threshold,
    nn.Softmin, nn.Softmax, nn.LogSoftmax, nn.Dropout, nn.AlphaDropout,
]

# NestedCollection = Union[T, Collection['NestedCollection']]


@lru_cache(maxsize=None)
def _no_map_type(container_type: Type[T]) -> Type[T]:
    # Create a subtype of the container type that sets an normally inaccessible
    # special attribute on instances.
    # This is necessary because `setattr` does not work on built-in types
    # (e.g. `list`).
    new_type = type("_no_map" + container_type.__name__,
                    (container_type,), {'--no-map--': True})
    return new_type


def no_map(container_type: Type[T], *args, **kwargs) -> T:
    r"""Create a "`non-mappable`" container type, i.e. it will be treated as a
    singleton object in :meth:`map_structure` and :meth:`map_structure_zip`,
    its contents will not be traversed.

    This is implemented by dynamically creating a subclass of the required type,
    and overriding the :attr:`__subclasscheck__` class method to always return
    `False`.

    Args:
        container_type: The type of the container to create,
            e.g. `list`, `dict`.
        args: Arguments to the constructor.
        kwargs: Keyword arguments to the constructor

    Returns:
        The `non-mappable` container type.
    """
    return _no_map_type(container_type)(*args, **kwargs)  # type: ignore


@no_type_check
def map_structure(fn: Callable[[T], R], obj: Collection[T]) -> Collection[R]:
    r"""Map a function over all elements in a (possibly nested) collection.

    Args:
        fn (callable): The function to call on elements.
        obj: The collection to map function over.

    Returns:
        The collection in the same structure, with elements mapped.
    """
    if hasattr(obj, "--no-map--"):
        return fn(obj)
    if isinstance(obj, list):
        return [map_structure(fn, x) for x in obj]
    if isinstance(obj, tuple):
        if isinstance(obj, torch.Size):
            return fn(obj)
        if hasattr(obj, '_fields'):  # namedtuple
            return type(obj)(*[map_structure(fn, x) for x in obj])
        else:
            return tuple(map_structure(fn, x) for x in obj)
    if isinstance(obj, dict):
        return {k: map_structure(fn, v) for k, v in obj.items()}
    if isinstance(obj, set):
        return {map_structure(fn, x) for x in obj}
    return fn(obj)


@no_type_check
def map_structure_zip(fn: Callable[..., R],
                      objs: Sequence[Collection[T]]) -> Collection[R]:
    r"""Map a function over tuples formed by taking one elements from each
    (possibly nested) collection. Each collection must have identical
    structures.

    .. note::
        Although identical structures are required, it is not enforced by
        assertions. The structure of the first collection is assumed to be
        the structure for all collections.

        For rare cases where collections need to have different structures,
        refer to :meth:`no_map`.

    Args:
        fn (callable): The function to call on elements.
        objs: The list of collections to map function over.

    Returns:
        A collection with the same structure, with elements mapped.
    """
    obj = objs[0]
    if hasattr(obj, "--no-map--"):
        return fn(*objs)
    if isinstance(obj, list):
        return [map_structure_zip(fn, xs) for xs in zip(*objs)]
    if isinstance(obj, tuple):
        if isinstance(obj, torch.Size):
            return fn(obj)
        if hasattr(obj, '_fields'):  # namedtuple
            return type(obj)(*[map_structure_zip(fn, xs) for xs in zip(*objs)])
        else:
            return tuple(map_structure_zip(fn, xs) for xs in zip(*objs))
    if isinstance(obj, dict):
        return {k: map_structure_zip(fn, [o[k] for o in objs])
                for k in obj.keys()}
    if isinstance(obj, set):
        return {map_structure_zip(fn, xs) for xs in zip(*objs)}
    return fn(*objs)


def get_first_in_structure(obj: Collection[T]) -> Optional[T]:
    r"""Return the first not-`None` element within a (possibly nested)
    collection.

    Args:
        obj: The collection to pick the element from.

    Returns:
        The first non-`None` element from the collection, or `None` if the
        collection is empty or contains only `None` elements.
    """
    item = None

    def _get_first(x):
        nonlocal item
        if item is None:
            item = x

    map_structure(_get_first, obj)
    return item


def sequence_mask(lengths: Union[torch.LongTensor, List[int]],
                  max_len: Optional[int] = None,
                  dtype: Optional[torch.dtype] = None,
                  device: Optional[torch.device] = None) -> torch.ByteTensor:
    r"""Return a mask tensor representing the first N positions of each cell.

    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with

    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```

    Examples:

    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]

    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```

    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    row_vector = row_vector
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask


def get_args(fn: Callable) -> List[str]:
    r"""Gets the arguments of a function.

    Args:
        fn (callable): The function to inspect.

    Returns:
        list: A list of argument names (``str``) of the function.
    """
    argspec = inspect.getfullargspec(fn)
    args = argspec.args

    # Empty args can be because `fn` is decorated. Use `funcsigs.signature`
    # to re-do the inspect
    if len(args) == 0:
        args = funcsigs.signature(fn).parameters.keys()
        args = list(args)

    return args


def get_default_arg_values(fn: Callable) -> Dict[str, Any]:
    r"""Gets the arguments and respective default values of a function.

    Only arguments with default values are included in the output dictionary.

    Args:
        fn (callable): The function to inspect.

    Returns:
        dict: A dictionary that maps argument names (``str``) to their default
        values. The dictionary is empty if no arguments have default values.
    """
    argspec = inspect.getfullargspec(fn)
    if argspec.defaults is None:
        return {}
    num_defaults = len(argspec.defaults)
    return dict(zip(argspec.args[-num_defaults:], argspec.defaults))


def check_or_get_class(class_or_name: Union[Type[T], str],
                       module_paths: Optional[List[str]] = None,
                       superclass: Optional[MaybeTuple[type]] = None) \
        -> Type[T]:
    r"""Returns the class and checks if the class inherits :attr:`superclass`.

    Args:
        class_or_name: Name or full path to the class, or the class itself.
        module_paths (list, optional): Paths to candidate modules to search
            for the class. This is used if :attr:`class_or_name` is a string and
            the class cannot be located solely based on :attr:`class_or_name`.
            The first module in the list that contains the class
            is used.
        superclass (optional): A (list of) classes that the target class
            must inherit.

    Returns:
        The target class.

    Raises:
        ValueError: If class is not found based on :attr:`class_or_name` and
            :attr:`module_paths`.
        TypeError: If class does not inherits :attr:`superclass`.
    """
    class_ = class_or_name
    if isinstance(class_, str):
        class_ = get_class(class_, module_paths)
    if superclass is not None:
        if not issubclass(class_, superclass):
            raise TypeError(
                "A subclass of {} is expected. Got: {}".format(
                    superclass, class_))
    return class_


def get_class(class_name: str,
              module_paths: Optional[List[str]] = None) -> type:
    r"""Returns the class based on class name.

    Args:
        class_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            ``class_name``. The first module in the list that contains the class
            is used.

    Returns:
        The target class.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
    """
    class_ = locate(class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            class_ = locate('.'.join([module_path, class_name]))
            if class_ is not None:
                break

    if class_ is None:
        raise ValueError(
            "Class not found in {}: {}".format(module_paths, class_name))

    return class_  # type: ignore


def check_or_get_instance(ins_or_class_or_name: Union[Type[T], T, str],
                          kwargs: Kwargs,
                          module_paths: Optional[List[str]] = None,
                          classtype: Optional[MaybeTuple[type]] = None) -> T:
    r"""Returns a class instance and checks types.

    Args:
        ins_or_class_or_name: Can be of 3 types:

            - A class to instantiate.
            - A string of the name or full path to a class to instantiate.
            - The class instance to check types.

        kwargs (dict): Keyword arguments for the class constructor. Ignored
            if ``ins_or_class_or_name`` is a class instance.
        module_paths (list, optional): Paths to candidate modules to
            search for the class. This is used if the class cannot be
            located solely based on :attr:`class_name`. The first module
            in the list that contains the class is used.
        classtype (optional): A (list of) class of which the instance must
            be an instantiation.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
        ValueError: If :attr:`kwargs` contains arguments that are invalid
            for the class construction.
        TypeError: If the instance is not an instantiation of
            :attr:`classtype`.
    """
    ret = ins_or_class_or_name
    if isinstance(ret, (str, type)):
        ret = get_instance(ret, kwargs, module_paths)
    if classtype is not None:
        if not isinstance(ret, classtype):
            raise TypeError(
                "An instance of {} is expected. Got: {}".format(classtype, ret))
    return ret


def get_instance(class_or_name: Union[Type[T], str], kwargs: Optional[Kwargs],
                 module_paths: Optional[List[str]] = None) -> T:
    r"""Creates a class instance.

    Args:
        class_or_name: A class, or its name or full path to a class to
            instantiate.
        kwargs (dict): Keyword arguments for the class constructor.
        module_paths (list, optional): Paths to candidate modules to
            search for the class. This is used if the class cannot be
            located solely based on :attr:`class_name`. The first module
            in the list that contains the class is used.

    Returns:
        A class instance.

    Raises:
        ValueError: If class is not found based on :attr:`class_or_name` and
            :attr:`module_paths`.
        ValueError: If :attr:`kwargs` contains arguments that are invalid
            for the class construction.
    """
    # Locate the class
    class_ = class_or_name
    if isinstance(class_, str):
        class_ = get_class(class_, module_paths)

    # Check validity of arguments
    class_args = set(get_args(class_.__init__))

    if kwargs is None:
        kwargs = {}
    for key in kwargs.keys():
        if key not in class_args:
            raise ValueError(
                "Invalid argument for class %s.%s: %s, valid args: %s" %
                (class_.__module__, class_.__name__, key, list(class_args)))

    return class_(**kwargs)  # type: ignore


def check_or_get_instance_with_redundant_kwargs(
        ins_or_class_or_name: Union[Type[T], T, str], kwargs: Kwargs,
        module_paths: Optional[List[str]] = None,
        classtype: Optional[Type[T]] = None) -> T:
    r"""Returns a class instance and checks types.

    Only those keyword arguments in :attr:`kwargs` that are included in the
    class construction method are used.

    Args:
        ins_or_class_or_name: Can be of 3 types:

            - A class to instantiate.
            - A string of the name or module path to a class to instantiate.
            - The class instance to check types.

        kwargs (dict): Keyword arguments for the class constructor.
        module_paths (list, optional): Paths to candidate modules to
            search for the class. This is used if the class cannot be
            located solely based on :attr:`class_name`. The first module
            in the list that contains the class is used.
        classtype (optional): A (list of) classes of which the instance must
            be an instantiation.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
        ValueError: If :attr:`kwargs` contains arguments that are invalid
            for the class construction.
        TypeError: If the instance is not an instantiation of
            :attr:`classtype`.
    """
    ret = ins_or_class_or_name
    if isinstance(ret, (str, type)):
        ret = get_instance_with_redundant_kwargs(ret, kwargs, module_paths)
    if classtype is not None:
        if not isinstance(ret, classtype):
            raise TypeError(
                "An instance of {} is expected. Got: {}".format(classtype, ret))
    return ret


def get_instance_with_redundant_kwargs(
        class_name: Union[Type[T], str], kwargs: Kwargs,
        module_paths: Optional[List[str]] = None) -> T:
    r"""Creates a class instance.

    Only those keyword arguments in :attr:`kwargs` that are included in the
    class construction method are used.

    Args:
        class_name (str): A class or its name or module path.
        kwargs (dict): A dictionary of arguments for the class constructor. It
            may include invalid arguments which will be ignored.
        module_paths (list of str): A list of paths to candidate modules to
            search for the class. This is used if the class cannot be located
            solely based on :attr:`class_name`. The first module in the list
            that contains the class is used.

    Returns:
        A class instance.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
    """
    # Locate the class
    if isinstance(class_name, str):
        class_ = get_class(class_name, module_paths)
    else:
        class_ = class_name

    # Select valid arguments
    selected_kwargs = {}
    class_args = set(get_args(class_.__init__))  # type: ignore
    if kwargs is None:
        kwargs = {}
    for key, value in kwargs.items():
        if key in class_args:
            selected_kwargs[key] = value

    return class_(**selected_kwargs)


def get_function(fn_or_name: Union[str, Callable[[torch.Tensor], torch.Tensor]],
                 module_paths: Optional[List[str]] = None) \
        -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Returns the function of specified name and module.

    Args:
        fn_or_name (str or callable): Name or full path to a function, or the
            function itself.
        module_paths (list, optional): A list of paths to candidate modules to
            search for the function. This is used only when the function
            cannot be located solely based on :attr:`fn_or_name`. The first
            module in the list that contains the function is used.

    Returns:
        A function.

    Raises:
        ValueError: If method with name as :attr:`fn_or_name` is not found.
    """
    if callable(fn_or_name):
        return fn_or_name

    fn = locate(fn_or_name)
    if (fn is None) and (module_paths is not None):
        for module_path in module_paths:
            fn = locate('.'.join([module_path, fn_or_name]))
            if fn is not None:
                break

    if fn is None:
        raise ValueError(
            "Method not found in {}: {}".format(module_paths, fn_or_name))

    return fn  # type: ignore


def call_function_with_redundant_kwargs(fn: Callable[..., R],
                                        kwargs: Dict[str, Any]) -> R:
    r"""Calls a function and returns the results.

    Only those keyword arguments in :attr:`kwargs` that are included in the
    function's argument list are used to call the function.

    Args:
        fn (function): A callable. If :attr:`fn` is not a python function,
            :attr:`fn.__call__` is called.
        kwargs (dict): A ``dict`` of arguments for the callable. It
            may include invalid arguments which will be ignored.

    Returns:
        The returned results by calling :attr:`fn`.
    """
    try:
        fn_args = set(get_args(fn))
    except TypeError:
        fn_args = set(get_args(fn.__call__))  # type: ignore

    if kwargs is None:
        kwargs = {}

    # Select valid arguments
    selected_kwargs = {}
    for key, value in kwargs.items():
        if key in fn_args:
            selected_kwargs[key] = value

    return fn(**selected_kwargs)


def get_instance_kwargs(kwargs: Kwargs, hparams: ParamDict) -> Kwargs:
    r"""Makes a dictionary of keyword arguments with the following structure:

    ``kwargs_ = {'hparams': dict(hparams), **kwargs}``.

    This is typically used for constructing a module which takes a set of
    arguments as well as a argument named ``"hparams"``.

    Args:
        kwargs (dict): A ``dict`` of keyword arguments. Can be `None`.
        hparams: A ``dict`` or an instance of :class:`~texar.torch.HParams`.
            Can be `None`.

    Returns:
        A ``dict`` that contains the keyword arguments in :attr:`kwargs`, and
        an additional keyword argument named ``"hparams"``.
    """
    if hparams is None or isinstance(hparams, dict):
        kwargs_ = {'hparams': hparams}
    elif isinstance(hparams, HParams):
        kwargs_ = {'hparams': hparams.todict()}
    else:
        raise ValueError(
            '`hparams` must be a dict, an instance of HParams, or a `None`.')
    kwargs_.update(kwargs or {})
    return kwargs_


def get_output_size(input_instance: nn.Module) -> Optional[int]:
    r"""Return the final dimension size of :attr:`input_instance` output.

    If type of :attr:`input_instance` is among the common types, the final
    dimension size will be computed.

    Args:
        input_instance: A :class:`~torch.nn.Module` instance from
            which to compute the final dimension size.

    Returns:
        int (optional): The final dimension size of the output.
            If output size is determined by input, returns ``-1``,
            otherwise if output size is not computable, return `None`.
    """

    for t, l in Type_size_lambda_map.items():
        if isinstance(input_instance, t):
            return l(input_instance)
    for t in Type_size_keeper:
        if isinstance(input_instance, t):
            return -1
    return None


def dict_patch(tgt_dict: AnyDict, src_dict: AnyDict) -> AnyDict:
    r"""Recursively patch :attr:`tgt_dict` by adding items from :attr:`src_dict`
    that do not exist in :attr:`tgt_dict`.

    If respective items in :attr:`src_dict` and :attr:`tgt_dict` are both
    ``dict``, the :attr:`tgt_dict` item is patched recursively.

    Args:
        tgt_dict (dict): Target dictionary to patch.
        src_dict (dict): Source dictionary.

    Returns:
        dict: The new :attr:`tgt_dict` that is patched.
    """
    if src_dict is None:
        return tgt_dict

    for key, value in src_dict.items():
        if key not in tgt_dict:
            tgt_dict[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(tgt_dict[key], dict):
            tgt_dict[key] = dict_patch(tgt_dict[key], value)
    return tgt_dict


def dict_lookup(dict_: MutableMapping[K, V], keys: Union[List[K], np.ndarray],
                default: Optional[V] = None) -> np.ndarray:
    r"""Looks up :attr:`keys` in the dictionary, returns the corresponding
    values.

    The :attr:`default` is used for keys not present in the dictionary.

    Args:
        dict\_ (dict): A dictionary for lookup.
        keys: A numpy array or a (possibly nested) list of keys.
        default (optional): Value to be returned when a key is not in
            :attr:`dict_`. Error is raised if :attr:`default` is not given and
            key is not in the dictionary.

    Returns:
        A numpy array of values with the same structure as :attr:`keys`.

    Raises:
        TypeError: If key is not in :attr:`dict_` and :attr:`default` is
            `None`.
    """
    return np.vectorize(lambda x: dict_.get(x, default))(keys)  # type: ignore


# TODO: Remove these once pylint supports function stubs.
# pylint: disable=unused-argument,function-redefined,multiple-statements

@overload
def dict_fetch(src_dict: ParamDict,
               tgt_dict_or_keys: Union[ParamDict, List[str]]) -> AnyDict: ...


@overload
def dict_fetch(src_dict: None, tgt_dict_or_keys: Any) -> None: ...


def dict_fetch(src_dict: Optional[ParamDict],
               tgt_dict_or_keys: Union[ParamDict, List[str]]) \
        -> Optional[AnyDict]:
    r"""Fetches a sub-dictionary of :attr:`src_dict` with the keys in
    :attr:`tgt_dict_or_keys`.

    Args:
        src_dict: A dictionary or instance of :class:`~texar.torch.HParams`.
            The source dictionary to fetch values from.
        tgt_dict_or_keys: A dictionary, instance of
            :class:`~texar.torch.HParams`, or a list (or a
            ``dict_keys``/``KeysView``) of keys to be included in the output
            dictionary.

    Returns:
        A new dictionary that is a sub-dictionary of :attr:`src_dict`.
    """
    if src_dict is None:
        return src_dict

    if isinstance(tgt_dict_or_keys, HParams):
        tgt_dict_or_keys = tgt_dict_or_keys.todict()
    if isinstance(tgt_dict_or_keys, MutableMapping):
        tgt_dict_or_keys = tgt_dict_or_keys.keys()  # type: ignore
    keys = list(tgt_dict_or_keys)

    if isinstance(src_dict, HParams):
        src_dict = src_dict.todict()

    return {k: src_dict[k] for k in keys if k in src_dict}


# pylint: enable=unused-argument,function-redefined,multiple-statements


def dict_pop(dict_: MutableMapping[T, Any], pop_keys: MaybeSeq[T],
             default: Optional[Any] = None) -> Dict[T, Any]:
    r"""Removes keys from a dictionary and returns their values.

    Args:
        dict\_ (dict): A dictionary from which items are removed.
        pop_keys: A key or a list of keys to remove and return respective
            values or :attr:`default`.
        default (optional): Value to be returned when a key is not in
            :attr:`dict_`. The default value is `None`.

    Returns:
        A ``dict`` of the items removed from :attr:`dict_`.
    """
    if not isinstance(pop_keys, (list, tuple)):
        pop_keys = cast(List[T], [pop_keys])
    ret_dict = {key: dict_.pop(key, default) for key in pop_keys}
    return ret_dict


def flatten_dict(dict_: AnyDict, parent_key: str = "", sep: str = "."):
    r"""Flattens a nested dictionary. Namedtuples within the dictionary are
    also converted to dictionaries.

    Adapted from:
    https://github.com/google/seq2seq/blob/master/seq2seq/models/model_base.py

    Args:
        dict\_ (dict): The dictionary to flatten.
        parent_key (str): A prefix to prepend to each key.
        sep (str): Separator that intervenes between parent and child keys.
            For example, if :attr:`sep` == ``"."``, then ``{ "a": { "b": 3 } }``
            is converted into ``{ "a.b": 3 }``.

    Returns:
        A new flattened ``dict``.
    """
    items: List[Tuple[str, Any]] = []
    for key, value in dict_.items():
        key_ = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, key_, sep=sep).items())
        elif isinstance(value, tuple) and hasattr(value, '_asdict'):
            # namedtuple
            dict_items = collections.OrderedDict(
                zip(value._fields, value))  # type: ignore
            items.extend(flatten_dict(dict_items, key_, sep=sep).items())
        else:
            items.append((key_, value))
    return dict(items)


def default_str(str_: Optional[str], default: str) -> str:
    r"""Returns :attr:`str_` if it is not `None` or empty, otherwise returns
    :attr:`default_str`.

    Args:
        str\_: A string.
        default: A string.

    Returns:
        Either :attr:`str_` or :attr:`default_str`.
    """
    if str_ is not None and str_ != "":
        return str_
    else:
        return default


def uniquify_str(str_: str, str_set: Collection[str]) -> str:
    r"""Uniquifies :attr:`str_` if :attr:`str_` is included in :attr:`str_set`.

    This is done by appending a number to :attr:`str_`. Returns
    :attr:`str_` directly if it is not included in :attr:`str_set`.

    Args:
        str\_ (string): A string to uniquify.
        str_set (set, dict, or list): A collection of strings. The returned
            string is guaranteed to be different from the elements in the
            collection.

    Returns:
        The uniquified string. Returns :attr:`str_` directly if it is
        already unique.

    Example:

        .. code-block:: python

            print(uniquify_str('name', ['name', 'name_1']))
            # 'name_2'

    """
    if str_ not in str_set:
        return str_
    else:
        for i in range(1, len(str_set) + 1):
            unique_str = str_ + "_%d" % i
            if unique_str not in str_set:
                return unique_str
    raise ValueError("Failed to uniquify string: " + str_)


def _recur_split(s: MaybeSeq[str],
                 dtype_as: Collection[str]) -> MaybeSeq[str]:
    r"""Splits (possibly nested list of) strings recursively.
    """
    if isinstance(s, str):
        return _maybe_list_to_array(s.split(), dtype_as)
    else:
        s_ = [_recur_split(si, dtype_as) for si in s]
        return _maybe_list_to_array(s_, s)


def strip_token(str_: MaybeSeq[str], token: str,
                is_token_list: bool = False) -> MaybeSeq[str]:
    r"""Returns a copy of strings with leading and trailing tokens removed.

    Note that besides :attr:`token`, all leading and trailing whitespace
    characters are also removed.

    If :attr:`is_token_list` is False, then the function assumes tokens in
    :attr:`str_` are separated with whitespace character.

    Args:
        str\_: A ``str``, or an ``n``-D numpy array or (possibly nested)
            list of ``str``.
        token (str): The token to strip, e.g., the ``"<PAD>"`` token defined in
            :class:`~texar.torch.data.SpecialTokens`.
        is_token_list (bool): Whether each sentence in :attr:`str_` is a list
            of tokens. If False, each sentence in :attr:`str_` is assumed to
            contain tokens separated with space character.

    Returns:
        The stripped strings of the same structure/shape as :attr:`str_`.

    Example:

        .. code-block:: python

            str_ = '<PAD> a sentence <PAD> <PAD>  '
            str_stripped = strip_token(str_, '<PAD>')
            # str_stripped == 'a sentence'

            str_ = ['<PAD>', 'a', 'sentence', '<PAD>', '<PAD>', '', '']
            str_stripped = strip_token(str_, '<PAD>', is_token_list=True)
            # str_stripped == 'a sentence'
    """

    def _recur_strip(s):
        if isinstance(s, str):
            if token == "":
                return ' '.join(s.strip().split())
            else:
                return (' '.join(s.strip().split())
                        .replace(' ' + token, '').replace(token + ' ', ''))
        else:
            s_ = [_recur_strip(si) for si in s]
            return _maybe_list_to_array(s_, s)

    s = str_

    if is_token_list:
        s = str_join(s)  # type: ignore

    strp_str = _recur_strip(s)

    if is_token_list:
        strp_str = _recur_split(strp_str, str_)

    return strp_str


def strip_eos(str_: MaybeSeq[str], eos_token: str = '<EOS>',
              is_token_list: bool = False) -> MaybeSeq[str]:
    r"""Remove the EOS token and all subsequent tokens.

    If :attr:`is_token_list` is False, then the function assumes tokens in
    :attr:`str_` are separated with whitespace character.

    Args:
        str\_: A ``str``, or an ``n``-D numpy array or (possibly nested)
            list of ``str``.
        eos_token (str): The EOS token. Default is ``"<EOS>"`` as defined in
            :class:`~texar.torch.data.SpecialTokens`.EOS
        is_token_list (bool): Whether each sentence in :attr:`str_` is a list
            of tokens. If False, each sentence in :attr:`str_` is assumed to
            contain tokens separated with space character.

    Returns:
        Strings of the same structure/shape as :attr:`str_`.
    """

    def _recur_strip(s):
        if isinstance(s, str):
            s_tokens = s.split()
            if eos_token in s_tokens:
                return ' '.join(s_tokens[:s_tokens.index(eos_token)])
            else:
                return s
        else:
            s_ = [_recur_strip(si) for si in s]
            return _maybe_list_to_array(s_, s)

    s = str_

    if is_token_list:
        s = str_join(s)  # type: ignore

    strp_str = _recur_strip(s)

    if is_token_list:
        strp_str = _recur_split(strp_str, str_)

    return strp_str


_strip_eos_ = strip_eos


def strip_bos(str_: MaybeSeq[str], bos_token: str = '<BOS>',
              is_token_list: bool = False) -> MaybeSeq[str]:
    r"""Remove all leading BOS tokens.

    Note that besides :attr:`bos_token`, all leading and trailing whitespace
    characters are also removed.

    If :attr:`is_token_list` is False, then the function assumes tokens in
    :attr:`str_` are separated with whitespace character.

    Args:
        str_: A ``str``, or an ``n``-D numpy array or (possibly nested)
            list of ``str``.
        bos_token (str): The BOS token. Default is ``"<BOS>"`` as defined in
            :class:`~texar.torch.data.SpecialTokens`.BOS
        is_token_list (bool): Whether each sentence in :attr:`str_` is a list
            of tokens. If False, each sentence in :attr:`str_` is assumed to
            contain tokens separated with space character.

    Returns:
        Strings of the same structure/shape as :attr:`str_`.
    """

    def _recur_strip(s):
        if isinstance(s, str):
            if bos_token == '':
                return ' '.join(s.strip().split())
            else:
                return ' '.join(s.strip().split()).replace(bos_token + ' ', '')
        else:
            s_ = [_recur_strip(si) for si in s]
            return _maybe_list_to_array(s_, s)

    s = str_

    if is_token_list:
        s = str_join(s)  # type: ignore

    strp_str = _recur_strip(s)

    if is_token_list:
        strp_str = _recur_split(strp_str, str_)

    return strp_str


_strip_bos_ = strip_bos


# pylint: disable=redefined-outer-name

def strip_special_tokens(str_: MaybeSeq[str],
                         strip_pad: Optional[str] = '<PAD>',
                         strip_bos: Optional[str] = '<BOS>',
                         strip_eos: Optional[str] = '<EOS>',
                         is_token_list: bool = False) -> MaybeSeq[str]:
    r"""Removes special tokens in strings, including:

        - Removes EOS and all subsequent tokens
        - Removes leading and and trailing PAD tokens
        - Removes leading BOS tokens

    Note that besides the special tokens, all leading and trailing whitespace
    characters are also removed.

    This is a joint function of :func:`strip_eos`, :func:`strip_pad`, and
    :func:`strip_bos`

    Args:
        str\_: A ``str``, or an ``n``-D numpy array or (possibly nested)
            list of ``str``.
        strip_pad (str): The PAD token to strip from the strings (i.e., remove
            the leading and trailing PAD tokens of the strings). Default
            is ``"<PAD>"`` as defined in
            :class:`~texar.torch.data.SpecialTokens`.PAD.
            Set to `None` or `False` to disable the stripping.
        strip_bos (str): The BOS token to strip from the strings (i.e., remove
            the leading BOS tokens of the strings).
            Default is ``"<BOS>"`` as defined in
            :class:`~texar.torch.data.SpecialTokens`.BOS.
            Set to `None` or `False` to disable the stripping.
        strip_eos (str): The EOS token to strip from the strings (i.e., remove
            the EOS tokens and all subsequent tokens of the strings).
            Default is ``"<EOS>"`` as defined in
            :class:`~texar.torch.data.SpecialTokens`.EOS.
            Set to `None` or `False` to disable the stripping.
        is_token_list (bool): Whether each sentence in :attr:`str_` is a list
            of tokens. If `False`, each sentence in :attr:`str_` is assumed to
            contain tokens separated with space character.

    Returns:
        Strings of the same shape of :attr:`str_` with special tokens stripped.
    """
    s = str_

    if is_token_list:
        s = str_join(s)  # type: ignore

    if strip_eos is not None and strip_eos is not False:
        s = _strip_eos_(s, strip_eos, is_token_list=False)

    if strip_pad is not None and strip_pad is not False:
        s = strip_token(s, strip_pad, is_token_list=False)

    if strip_bos is not None and strip_bos is not False:
        s = _strip_bos_(s, strip_bos, is_token_list=False)

    if is_token_list:
        s = _recur_split(s, str_)

    return s


def str_join(tokens: Sequence[List], sep: str = ' ') -> Sequence[str]:
    r"""Concatenates :attr:`tokens` along the last dimension with intervening
    occurrences of :attr:`sep`.

    Args:
        tokens: An ``n``-D numpy array or (possibly nested) list of ``str``.
        sep (str): The string intervening between the tokens.

    Returns:
        An ``(n-1)``-D numpy array (or list) of ``str``.
    """

    def _recur_join(s):
        if len(s) == 0:
            return ''
        elif isinstance(s[0], str):
            return sep.join(s)
        else:
            s_ = [_recur_join(si) for si in s]
            return _maybe_list_to_array(s_, s)

    str_ = _recur_join(tokens)

    return str_


# pylint: enable=redefined-outer-name

def ceildiv(a: int, b: int) -> int:
    r"""Compute division with results rounding up.

    For example, ``5 / 2 = 2.5``, ``ceildiv(5, 2) = 3``.

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        int: The quotient, rounded up.
    """
    return -(-a // b)


def sum_tensors(xs: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    r"""Sum a list of tensors with possible `None` values.

    Args:
        xs: A list of tensors.

    Returns:
        The summation of all the elements in the list.
    """
    idx = next((idx for idx, tensor in enumerate(xs) if tensor is not None), -1)
    if idx == -1:
        return None
    ret = xs[idx]
    for tensor in xs[(idx + 1):]:
        if tensor is not None:
            ret = ret + tensor
    return ret


def truncate_seq_pair(tokens_a: Union[List[int], List[str]],
                      tokens_b: Union[List[int], List[str]],
                      max_length: int):
    r"""Truncates a sequence pair in place to the maximum length.

    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal
    percent of tokens from each, since if one sequence is very short then
    each token that's truncated likely contains more information than a
    longer sequence.

    Example:
        tokens_a = [1, 2, 3, 4, 5]
        tokens_b = [6, 7]
        truncate_seq_pair(tokens_a, tokens_b, 5)
        tokens_a  # [1, 2, 3]
        tokens_b  # [6, 7]

    Args:
        tokens_a: A list of tokens or token ids.
        tokens_b: A list of tokens or token ids.
        max_length: maximum sequence length.
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
