from functools import partial, reduce
from operator import itemgetter
from typing import Dict, Sequence, Tuple, Union

NamedDimMap = Dict[str, Union[int, Sequence[int]]]


class ShapeDef(Tuple[Union[str, int], ...]):
    pass


def map_nested(f, data, *other_data, stop_type=None):
    if stop_type and type(data) == stop_type:
        return f(data, *other_data)
    elif isinstance(data, dict):
        gen = ((k, (data[k], *map(itemgetter(k), other_data))) for k in data)
        return type(data)((k, map_nested(f, *v, stop_type=stop_type)) for k, v in gen)
    elif isinstance(data, (tuple, list, set)):
        def gen():  # yapf: disable
            return (map_nested(f, x, *oth_x, stop_type=stop_type)
                    for i, (x, *oth_x) in enumerate(zip(data, *other_data)))

        try:
            return type(data)(gen())
        except TypeError:  # for namedtuples
            # TODO: use namedtuple names instead of indices
            return type(data)(*gen())
    else:
        return f(data, *other_data)


def reduce_nested(f, data, initial=None, stop_type=None):
    if stop_type and type(data) == stop_type:
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
