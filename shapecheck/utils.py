from functools import partial, reduce
from operator import itemgetter
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

T = TypeVar('T')
NestedStruct = Union[T, Any]
# NestedStruct = Union[NestedStruct[T], T, Dict[Any, T], List[T], Set[T], Tuple[T,...]]
NamedDimMap = Dict[str, Union[int, Tuple[int, ...]]]


class ShapeDef(Tuple[Union[str, int], ...]):
    pass


def map_nested(f: Callable,
               data: NestedStruct,
               *other_data: NestedStruct,
               stop_type: Optional[Type] = None) -> NestedStruct:
    # NOTE: Types should be:
    # (Callable[[T, ...], S], NestedStruct[T], NestedStruct, Optional[Type])
    # -> NestedStruct[S]
    if stop_type and isinstance(data, stop_type):
        return f(data, *other_data)
    elif isinstance(data, dict):
        gen = ((k, (data[k], *map(itemgetter(k), other_data))) for k in data)
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


def reduce_nested(f: Callable[[T, T], T],
                  data: NestedStruct[T],
                  initial: Optional[T] = None,
                  stop_type: Type = None) -> T:
    # NOTE: Types should be:
    # (Callable[[T, S], T], NestedStruct[S], Optional[T], Type) -> T
    if stop_type and isinstance(data, stop_type):
        return data
    elif isinstance(data, (dict, set, list, tuple)):
        it = data.values() if isinstance(data, dict) else data
        red_fn = partial(reduce_nested, stop_type=stop_type)
        if initial is None:
            return reduce(f, (red_fn(f, values) for values in it))
        else:
            return reduce(f, (red_fn(f, values) for values in it), initial)
    else:
        return data


def _green_highlight(s):
    return f'\x1b[6;30;42m{s}\x1b[0m'


def _red_highlight(s):
    return f'\x1b[6;30;41m{s}\x1b[0m'
