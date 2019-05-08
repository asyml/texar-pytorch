# Apply from Tensorflow(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/nest.py)

"""This module can perform operations on nested structures. A nested structure is a
Python sequence, tuple (including `namedtuple`), or dict that can contain
further sequences, tuples, and dicts.
"""

import collections
from collections import namedtuple, OrderedDict
from typing import TypeVar, Dict, Union, Callable, Tuple, List, Any
import torch


TypeArg = TypeVar('TypeArg')  # type argument
NestedStructure = Union[
    Dict[Any, "NestedStructure"],
    List["NestedStructure"],
    Tuple["NestedStructure"],
    TypeArg]

def is_sequence(seq: Union[List, Tuple]) -> bool:
    r"""If a instance is sequance(list, tuple, excluding torch.Size),
    return True, else False.
    Args:
        seq: instance to be checked.
    Returns:
        bool, True if the input instance is sequence, otherwise False.
    """
    if isinstance(seq, torch.Size):
        return False
    return isinstance(seq, (list, tuple))

def flatten(structure: NestedStructure) -> List:
    r"""Returns a flat list from a given nested structure.
    If nest is not a sequence, tuple, or dict, then returns a single-element
    list:[nest].
    In the case of dict instances, the sequence consists of the values,
    sorted by key to ensure deterministic behavior. This is true also for
    OrderedDict instances: their sequence order is ignored, the sorting order
    of keys is used instead. The same convention is followed in
    pack_sequence_as. This correctly repacks dicts and OrderedDicts after
    they have been flattened, and also allows flattening an OrderedDict
    and then repacking it back using a corresponding plain dict,
    or vice-versa. Dictionaries with non-sortable keys cannot be flattened.
    Users must not modify any collections used in nest while this function is
    running.
    Args:
        structure: an arbitrarily nested structure or a scalar object. Note,
        numpy arrays are considered scalars.
    Returns:
        A Python list, the flattened version of the input.
    Raises:
        TypeError: The nest is or contains a dict with non-sortable keys.
    """
    res = []
    if isinstance(structure, dict):
        structure = list(structure.values())
    if not is_sequence(structure):
        return [structure]
    else:
        for item in _yield_value(structure):
            res += flatten(item)
    return res

def pack_sequence_as(structure: NestedStructure,
                     flat_sequence: Union[List, Tuple]
                     ) -> NestedStructure:
    r"""Returns a given flattened sequence packed into a given structure.
    If `structure` is a scalar, `flat_sequence` must be a single-element list;
    in this case the return value is `flat_sequence[0]`.
    If `structure` is or contains a dict instance, the keys will be sorted to
    pack the flat sequence in deterministic order. This is true also for
    `OrderedDict` instances: their sequence order is ignored, the sorting
    order of keys is used instead. The same convention is followed in
    `flatten`. This correctly repacks dicts and `OrderedDict`s after they
    have been flattened, and also allows flattening an `OrderedDict` and
    then repacking it back using a corresponding plain dict, or vice-versa.
    Dictionaries with non-sortable keys cannot be flattened.
    Args:
        structure: Nested structure, whose structure is given by nested lists,
            tuples, and dicts. Note: numpy arrays and strings are considered
            scalars.
        flat_sequence: flat sequence to pack.
    Returns:
        packed: `flat_sequence` converted to have the same recursive
        structure as `structure`.
    Raises:
        ValueError: If `flat_sequence` and `structure` have different
        element counts.
        TypeError: `structure` is or contains a dict with non-sortable keys.
    """
    if not is_sequence(flat_sequence):
        raise TypeError("flat_sequence must be a sequence")
    if isinstance(structure, dict):
        structure = list(structure.values())
    if not is_sequence(structure):
        if len(flat_sequence) != 1:
            raise ValueError("Structure is a scalar"
                             "but len(flat_sequence) == %d > 1"
                             % len(flat_sequence))
        return flat_sequence[0]
    try:
        final_index, packed = _packed_nest_with_indices(structure,
                                                        flat_sequence, 0)
        if final_index < len(flat_sequence):
            raise IndexError
    except IndexError:
        flat_structure = flatten(structure)
        if len(flat_structure) != len(flat_sequence):
            raise ValueError(
                "Could not pack sequence. Structure had %d elements, but "
                "flat_sequence had %d elements.  Structure: %s,"
                "flat_sequence: %s." %
                (len(flat_structure),
                 len(flat_sequence),
                 structure,
                 flat_sequence))
    return _sequence_like(structure, packed)

def map_structure(func: Callable,
                  *structure: NestedStructure) -> NestedStructure:
    r"""Applies `func` to each entry in `structure` and returns a new
    structure. Applies `func(x[0], x[1], ...)` where x[i] is an entry in
    `structure[i]`.  All structures in `structure` must have the same arity,
    and the return value will contain results with the same structure layout.
    Args:
        func: A callable that accepts as many arguments as there are
        structures. *structure: scalar, or tuple or list of constructed
        scalars and/or other tuples/lists, or scalars.
        Note: numpy arrays are considered as scalars.
    Returns:
        A new structure with the same arity as `structure`, whose values
        correspond to `func(x[0], x[1], ...)` where `x[i]` is a value in the
        corresponding location in `structure[i]`.
    Raises:
        TypeError: If `func` is not callable or if the structures do not match
        each other by depth tree.
        ValueError: If no structure is provided or if the structures do not
        match each other by type.
        ValueError: If wrong keyword arguments are provided.
    """
    if not callable(func):
        raise TypeError("func must be callable, got: %s" % func)

    if not structure:
        raise ValueError("Must provide at least one structure")

    for other in structure[1:]:
        assert_same_structure(structure[0], other)

    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    return pack_sequence_as(
        structure[0], [func(*x) for x in entries])

def _assert_same_structure_helper(st1: NestedStructure,
                                  st2: NestedStructure):
    r"""Recursively check if two structures are nested in the same way.
    Helper for the `assert_same_structure`
    Args:
        st1: an arbitrarily nested structure.
        st2: an arbitrarily nested structure.
    Raises:
        ValueError: If the two structures do not have the same number of
        elements or if the two structures are not nested in the same way.
    """
    if is_sequence(st1) != is_sequence(st2):
        raise ValueError(
            "The two structures don't have the same nested structure.\n\n"
            "First structure: %s\n\nSecond structure: %s." % (st1, st2))

    if not is_sequence(st1):
        return

    st1_as_sequence = [n for n in _yield_value(st1)]
    st2_as_sequence = [n for n in _yield_value(st2)]
    for item1, item2 in zip(st1_as_sequence, st2_as_sequence):
        _assert_same_structure_helper(item1, item2)

def assert_same_structure(st1: NestedStructure,
                          st2: NestedStructure):
    r"""Asserts that two structures are nested in the same way.
    For instance, this code will print `True`:
    ```python
    def nt(a, b):
        return collections.namedtuple('foo', 'a b')(a, b)
    print(assert_same_structure(nt(0, 1), nt(2, 3)))
    ```
    Args:
        st1: an arbitrarily nested structure.
        st2: an arbitrarily nested structure.
    Raises:
        ValueError: If the two structures do not have the same number of
        elements or if the two structures are not nested in the same way.
    """
    len_st1 = 1
    if is_sequence(st1):
        len_st1 = len(flatten(st1))

    len_st2 = 1
    if is_sequence(st2):
        len_st2 = len(flatten(st2))

    if len_st1 != len_st2:
        raise ValueError("The two structures don't have the same number of "
                         "elements.\n\nFirst structure (%i elements): %s\n\n"
                         "Second structure (%i elements): %s"
                         % (len_st1, st1, len_st2, st2))
    _assert_same_structure_helper(st1, st2)

def _sorted(dict_: Dict[TypeArg, Any]) -> List[TypeArg]:
    r"""Returns a sorted list of the dict keys, with error if keys not
    sortable.
    """
    try:
        return sorted(dict_)
    except TypeError:
        raise TypeError("nest only supports dicts with sortable keys.")

def _is_namedtuple(instance: object) -> bool:
    r"""Returns True if `instance` is a `namedtuple`.
    Args:
        instance: An instance of a Python object.
    Returns:
        True if `instance` is a `namedtuple`.
    """
    type_ = type(instance)
    base_ = type_.__bases__
    if len(base_) != 1 or base_[0] != tuple:
        return False
    field_ = getattr(type_, '_fields', None)
    if not isinstance(field_, tuple):
        return False
    return all(isinstance(n, str) for n in field_)

def _yield_value(iterable):
    r"""Yield only sorted values from `iterable`.
    Args:
        iterable: an iterable.
    Yields:
        The iterable's values, in order of sorted keys or items.
    """
    for _, value in _yield_sorted_items(iterable):
        yield value

def _yield_sorted_items(iterable):
    r"""Yield (key, value) pairs for `iterable` in a deterministic order.
    For Sequences, the key will be an int, the array index of a value.
    For Mappings, the key will be the dictionary key.
    For objects (e.g. namedtuples), the key will be the attribute name.
    In all cases, the keys will be iterated in sorted order.
    Args:
        iterable: an iterable.
    Yields:
        The iterable's (key, value) pairs, in order of sorted keys.
    """
    if isinstance(iterable, collections.Mapping):
        for key in _sorted(iterable):
            yield key, iterable[key]

    elif _is_namedtuple(iterable):
        for field in iterable._fields:
            yield field, getattr(iterable, field)

    else:
        for item in enumerate(iterable):
            yield None, item

def _packed_nest_with_indices(structure: NestedStructure,
                              flat: NestedStructure,
                              index: int) -> Tuple[int, NestedStructure]:
    r"""Helper function for pack_sequence_as.
    Args:
        structure: Substructure (list / tuple / dict) to mimic.
        flat: Flattened values to output substructure for.
        index: Index at which to start reading from flat.
        is_seq: Function used to test if a value should be treated as a
            sequence.
    Returns:
        The tuple (new_index, child), where:
        * new_index - the updated index into `flat` having processed
                    `structure`.
        * packed - the subset of `flat` corresponding to `structure`,
                    having started at `index`, and packed into the same nested
                    format.
    Raises:
        ValueError: if `structure` contains more elements than `flat`
        (assuming indexing starts from `index`).
    """
    packed = []
    for value in _yield_value(structure):
        if is_sequence(value):
            new_index, child = _packed_nest_with_indices(value, flat, index)
            packed.append(_sequence_like(value, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed

InstanceType = Union[Tuple, List, Dict, namedtuple, OrderedDict]
def _sequence_like(instance: InstanceType,
                   args: NestedStructure) -> NestedStructure:
    r"""Converts the sequence `args` to the same type as `instance`.
    Args:
        instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,
            `collections.OrderedDict`.
        args: elements to be converted to the `instance` type.
    Returns:
        `args` with the type of `instance`.
    """
    if isinstance(instance, collections.Mapping):
        result = dict(zip(_sorted(instance), args))
        return type(instance)((key, result[key]) for key in instance)
    elif _is_namedtuple(instance):
        return type(instance)(*args)

    else:
        # Not a namedtuple
        return type(instance)(args)
