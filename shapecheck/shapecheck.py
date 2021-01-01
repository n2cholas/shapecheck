import functools
import inspect
from operator import and_, attrgetter
from typing import Any, Callable, Optional, Sequence

from .exception import ShapeError, _ShapeInfo
from .utils import NamedDimMap, ShapeDef, map_nested, reduce_nested


def is_compatible(shape: Sequence[int],
                  expected_shape: ShapeDef,
                  dim_dict: Optional[NamedDimMap] = None) -> bool:
    if dim_dict is None:
        dim_dict = {}

    has_ellipsis = any(
        isinstance(s, str) and s.endswith('...')
        for s in expected_shape)  # type: ignore
    if not has_ellipsis and len(shape) != len(expected_shape):
        return False
    if len(shape) < len(expected_shape) - int(has_ellipsis):
        return False

    s_ind = 0
    for exp_dim in expected_shape:
        exp_dim = dim_dict.get(exp_dim, exp_dim)  # type: ignore

        if exp_dim == -1:
            pass
        elif isinstance(exp_dim, int):
            if exp_dim != shape[s_ind]:
                return False
        elif isinstance(exp_dim, tuple):
            if exp_dim != shape[s_ind:s_ind + len(exp_dim)]:
                return False
            s_ind += len(exp_dim) - 1
        elif isinstance(exp_dim, str):
            if exp_dim.endswith('...'):
                diff = len(shape) - len(expected_shape)
                if exp_dim != '...':  # named variadic dimensions
                    dim_dict[exp_dim] = shape[s_ind:s_ind + diff + 1]
                s_ind += diff
            else:
                dim_dict[exp_dim] = shape[s_ind]

        s_ind += 1

    return s_ind >= len(shape)  # false when last named variadic dimensions don't match


def str_to_shape(string: Optional[str]) -> Optional[ShapeDef]:
    def gen():
        has_ellipsis = False
        for s in string.split(','):
            s = s.strip()
            if s.endswith('...'):
                if has_ellipsis:
                    raise RuntimeError('Each shape can have at most one \'...\'.'
                                       f'Got {string}')
                has_ellipsis = True
                yield s
            else:
                # need try-catch because str.isnumeric() misses -1
                try:
                    yield int(s)
                except ValueError:
                    yield s

    return ShapeDef(gen()) if string else None


def check_shapes(*in_shapes, out=None) -> Callable[[Callable], Callable]:
    in_shapes = map_nested(str_to_shape, in_shapes)
    out = map_nested(str_to_shape, out)

    def decorator(f: Callable) -> Callable:
        argspec = inspect.getfullargspec(f)
        full_argspec = argspec.args + argspec.kwonlyargs
        expected_shapes = dict(zip(full_argspec, in_shapes))

        @functools.wraps(f)
        def inner(*args: Any):
            assert len(args) == len(in_shapes)
            named_args = dict(zip(full_argspec, args))
            dim_dict: NamedDimMap = {}
            check_fn = functools.partial(_check_item, dim_dict=dim_dict)

            input_info = map_nested(check_fn,
                                    expected_shapes,
                                    named_args,
                                    stop_type=ShapeDef)
            nested_is_comp = map_nested(attrgetter('is_compatible'),
                                        input_info,
                                        stop_type=_ShapeInfo)
            if not reduce_nested(and_, nested_is_comp, initial=True):
                raise ShapeError(f.__name__, dim_dict, input_info)

            output = f(*args)

            output_info = map_nested(check_fn, out, output, stop_type=ShapeDef)
            nested_is_comp = map_nested(attrgetter('is_compatible'),
                                        output_info,
                                        stop_type=_ShapeInfo)
            if not reduce_nested(and_, nested_is_comp, initial=True):
                raise ShapeError(f.__name__, dim_dict, input_info, output_info)

            return output

        return inner

    return decorator


def _check_item(expected_shape: ShapeDef,
                arg: Any,
                dim_dict: Optional[NamedDimMap] = None) -> _ShapeInfo:
    if expected_shape is not None:
        is_comp = is_compatible(arg.shape, expected_shape, dim_dict)
        return _ShapeInfo(is_comp, expected_shape, arg.shape)
    else:
        return _ShapeInfo(True)
