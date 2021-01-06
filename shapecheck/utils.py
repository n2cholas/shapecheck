from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type, TypeVar, Union

NamedDimMap = Dict[str, Union[int, Tuple[int, ...]]]

T = TypeVar('T')
NestedStruct = Union[T, Any]
# NOTE: The definition below is more accurate, but requires recursive type-checking.
# NestedStruct = Union[NestedStruct[T], T, Dict[Any, T], List[T], Set[T], Tuple[T,...]]


class ShapeDef(Tuple[Union[str, int], ...]):
    """Tuple with dimensions and named (variadic) dimensions.

    We use a subclass to use more conveniently with map_nested, as we can use
    ShapeDef as the stop_type to apply functions to all the ShapeDefs in a
    structure.
    """


def map_nested(f: Callable,
               data: NestedStruct,
               *other_data: NestedStruct,
               stop_type: Optional[Type] = None) -> NestedStruct:
    """Apply f to ever element in the data (and other_data, if given).

    Note, signature should be:
        (Callable[[T, ...], S], NestedStruct[T], NestedStruct,
         Optional[Type]) -> NestedStruct[S]

    Args:
        f: callable to apply to ever element in data (excluding keys).
        data: nested dict/list/tuple/set of some value type.
        *other_data: specify additional nested structures which will be passed
            as additional arguments to f. The structure must be a superset of data.
            This means, for example, the corresponding dicts must have the keys in
            data, but could have additional keys.
        stop_type: container type at which to stop recursion. This is useful
            when, for example, you have a container subclass you want to apply f
            to.  If this wasn't specified, this function would apply f to the
            elements of that container subclass.

    Returns:
        Nested structure with the same structure as data which contains the result
        of applying f to each element of data (and the corresponding elements in
        other_data)
    """
    if stop_type and type(data) == stop_type:
        return f(data, *other_data)
    elif isinstance(data, dict):
        gen = ((k, (data[k], *(o[k] for o in other_data))) for k in data)
        return type(data)((k, map_nested(f, *v, stop_type=stop_type)) for k, v in gen)
    elif isinstance(data, (tuple, list, set)):
        gen = (map_nested(f, x, *oth_x, stop_type=stop_type)
               for x, *oth_x in zip(data, *other_data))
        try:
            return type(data)(gen)
        except TypeError:  # for namedtuples
            return type(data)(*gen)
    else:
        return f(data, *other_data)


def iterate_nested(data: NestedStruct, stop_type: Optional[Type] = None) -> Iterator:
    """Provide iterator for all (non-key) elements of data.

    Args:
        data: nested dict/set/list/tuple of values to be iterated over.
        stop_type: container type at which to stop recursion. This is useful
            when, for example, you have a container subclass nested in a
            dict/set/list/tuple that you want to yield from this function.

    Returns:
        Generator yielding each non-key element in data.
    """
    if stop_type and type(data) == stop_type:
        yield data
    elif isinstance(data, (dict, tuple, list, set)):
        for v in (data.values() if isinstance(data, dict) else data):
            yield from iterate_nested(v, stop_type=stop_type)
    else:
        yield data
