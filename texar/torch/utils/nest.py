# Apply from Tensorflow:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/nest.py

"""This module can perform operations on nested structures. A nested
structure is a Python sequence, tuple (including `namedtuple`), or
dict that can contain further sequences, tuples, and dicts.
"""

import collections
from typing import TypeVar, Mapping, Union, Tuple, List, Any
import torch


TypeArg = TypeVar('TypeArg')  # type argument
NestedStructure = Any


def is_sequence(seq: Any) -> bool:
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


def flatten(structure: NestedStructure) -> List[Any]:
    r"""Returns a flat list from a given nested structure.
    If nest is not a sequence, tuple, or dict, then returns a single-element
    `list:[nest]`.
    In the case of dict instances, the sequence consists of the values,
    sorted by key to ensure deterministic behavior. This is true also for
    `OrderedDict` instances: their sequence order is ignored, the sorting order
    of keys is used instead. The same convention is followed in
    :func:`~texar.torch.utils.nest.pack_sequence_as`. This correctly repacks
    dictionaries and `OrderedDict`s after they have been flattened, and also
    allows flattening an `OrderedDict` and then repacking it back using a
    corresponding plain dict, or vice-versa. Dictionaries with non-sortable
    keys cannot be flattened. Users must not modify any collections used in
    nest while this function is running.

    Args:
        structure: an arbitrarily nested structure or a scalar object. Note,
            numpy arrays are considered scalars.

    Returns:
        A Python list, the flattened version of the input.

    Raises:
        TypeError: The nest is or contains a dict with non-sortable keys.
    """
    res: List[Any] = []
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
    If ``structure`` is a scalar, ``flat_sequence`` must be a single-element
    list; in this case the return value is ``flat_sequence[0]``.
    If ``structure`` is or contains a dict instance, the keys will be sorted to
    pack the flat sequence in deterministic order. This is true also for
    `OrderedDict` instances: their sequence order is ignored, the sorting
    order of keys is used instead. The same convention is followed in
    :func:`~texar.torch.utils.nest.flatten`. This correctly repacks dictionaries
    and `OrderedDicts` after they have been flattened, and also allows
    flattening an `OrderedDict` and then repacking it back using a
    corresponding plain dict, or vice-versa. Dictionaries with non-sortable
    keys cannot be flattened.

    Args:
        structure: Nested structure, whose structure is given by nested lists,
            tuples, and dictionaries. Note: numpy arrays and strings are
            considered scalars.
        flat_sequence: flat sequence to pack.

    Returns:
        packed: ``flat_sequence`` converted to have the same recursive
        structure as ``structure``.

    Raises:
        ValueError: If ``flat_sequence`` and ``structure`` have different
            element counts.
        TypeError: ``structure`` is or contains a dict with non-sortable keys.
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


def _sorted(dict_: Mapping[Any, Any]) -> List[TypeArg]:
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
    if isinstance(iterable, collections.abc.Mapping):
        for key in _sorted(iterable):
            yield key, iterable[key]

    elif _is_namedtuple(iterable):
        for field in iterable._fields:
            yield field, getattr(iterable, field)

    else:
        for index, item in enumerate(iterable):
            yield index, item


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


InstanceType = Any


def _sequence_like(instance: InstanceType,
                   args: Any) -> InstanceType:
    r"""Converts the sequence `args` to the same type as `instance`.
    Args:
        instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,
            `collections.OrderedDict`.
        args: elements to be converted to the `instance` type.
    Returns:
        `args` with the type of `instance`.
    """
    if isinstance(instance, collections.abc.Mapping):
        result: Mapping[Any, Any] = dict(zip(_sorted(instance), args))
        generator = ((key, result[key]) for key in instance)
        return type(instance)(generator)  # type: ignore
    elif _is_namedtuple(instance):
        return type(instance)(*args)
    else:
        # Not a namedtuple
        return type(instance)(args)
