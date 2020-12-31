from functools import reduce
from operator import itemgetter


def map_nested(f, data, *other_data, stop_type=None):
    if stop_type and type(data) == stop_type:
        # stop the recursion at a particular container type
        return f(data, *other_data)
    elif isinstance(data, dict):
        gen = ((k, (data[k], *map(itemgetter(k), other_data))) for k in data)
        return type(data)((k, map_nested(f, *v, stop_type=stop_type)) for k, v in gen)
    elif isinstance(data, (tuple, list, set)):
        def gen():  # yapf: disable
            return (map_nested(f, x, *other_x, stop_type=stop_type)
                    for x, *other_x in zip(data, *other_data))

        try:
            return type(data)(gen())
        except TypeError:  # for namedtuple
            return type(data)(*gen())
    else:
        return f(data, *other_data)


def reduce_nested(f, data, initial=None):
    if isinstance(data, (dict, set, list, tuple)):
        it = data.values() if isinstance(data, dict) else data
        if initial is None:
            return reduce(f, (reduce_nested(f, values) for values in it))
        else:
            return reduce(f, (reduce_nested(f, values) for values in it), initial)
    else:
        return data


def _green_highlight(s):
    return f'\x1b[6;30;42m{s}\x1b[0m'


def _red_highlight(s):
    return f'\x1b[6;30;41m{s}\x1b[0m'
