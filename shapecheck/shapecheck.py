import functools
import inspect
from operator import and_, attrgetter
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union, cast

from .exception import ShapeError, _ShapeInfo
from .utils import NamedDimMap, NestedStruct, ShapeDef, map_nested, reduce_nested

__all__ = [
    'check_shapes', 'is_compatible', 'str_to_shape', 'set_checking_enabled',
    'is_checking_enabled'
]

# TODO: make these thread local
_CHECKING_ENABLED = True
_DIM_DICT: NamedDimMap = {}
# count number of functions in the call stack that have used each variable
_NAME_USE_CNT: Dict[str, int] = {}
_MATCH_CALLEES: bool = False


def is_compatible(shape: Tuple[int],
                  expected_shape: ShapeDef,
                  dim_dict: Optional[NamedDimMap] = None) -> bool:
    if dim_dict is None:
        dim_dict = {}

    has_ellipsis = any(isinstance(s, str) and s.endswith('...') for s in expected_shape)
    if not has_ellipsis and len(shape) != len(expected_shape):
        return False
    if len(shape) < len(expected_shape) - int(has_ellipsis):
        return False

    s_ind = 0
    for e in expected_shape:
        exp_dim = dim_dict.get(cast(str, e), e)

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


def check_shapes(*in_: NestedStruct[str],
                 out: Optional[NestedStruct[str]] = None,
                 match_callees: bool = False) -> Callable[[Callable], Callable]:
    in_shapes = cast(Tuple[NestedStruct[ShapeDef], ...], map_nested(str_to_shape, in_))
    out = map_nested(str_to_shape, out)

    def decorator(f: Callable) -> Callable:
        argspec = inspect.getfullargspec(f)
        full_argspec = argspec.args + argspec.kwonlyargs
        expected_shapes = dict(zip(full_argspec, in_shapes))

        @functools.wraps(f)
        def inner(*args: Any) -> Any:
            assert len(args) == len(in_shapes)

            if not _CHECKING_ENABLED:
                return f(*args)

            if ((match_callees or _MATCH_CALLEES)
                    and not inner.named_dim_set):  # type: ignore
                # TODO: get rid of map hack and fix reduce_nested
                collect_fn = functools.partial(
                    _collect_names, named_dim_set=inner.named_dim_set)  # type: ignore
                map_nested(collect_fn, (in_shapes, out))

            with _update_global_named_dim_info(inner.named_dim_set):  # type: ignore
                named_args = dict(zip(full_argspec, args))
                dim_dict: NamedDimMap = _DIM_DICT if match_callees or _DIM_DICT else {}
                check_fn = functools.partial(_check_item, dim_dict=dim_dict)

                input_info = cast(
                    Dict[str, NestedStruct[_ShapeInfo]],
                    map_nested(check_fn,
                               expected_shapes,
                               named_args,
                               stop_type=ShapeDef))
                nested_is_comp = map_nested(attrgetter('is_compatible'),
                                            input_info,
                                            stop_type=_ShapeInfo)
                if not reduce_nested(and_, nested_is_comp, initial=True):
                    raise ShapeError(f.__name__, dim_dict, input_info)

                with _match_callees_enabled(match_callees or _MATCH_CALLEES):
                    output = f(*args)

                output_info = map_nested(check_fn, out, output, stop_type=ShapeDef)
                nested_is_comp = map_nested(attrgetter('is_compatible'),
                                            output_info,
                                            stop_type=_ShapeInfo)
                if not reduce_nested(and_, nested_is_comp, initial=True):
                    raise ShapeError(f.__name__, dim_dict, input_info, output_info)

            return output

        inner.named_dim_set = set()  # type: ignore
        return inner

    return decorator


class set_checking_enabled:
    def __init__(self, mode: bool) -> None:
        global _CHECKING_ENABLED
        self.prev = _CHECKING_ENABLED
        _CHECKING_ENABLED = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _CHECKING_ENABLED
        _CHECKING_ENABLED = self.prev


def is_checking_enabled() -> bool:
    return _CHECKING_ENABLED


class _match_callees_enabled:
    def __init__(self, mode: bool) -> None:
        global _MATCH_CALLEES
        self.prev = _MATCH_CALLEES
        _MATCH_CALLEES = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _MATCH_CALLEES
        _MATCH_CALLEES = self.prev


class _update_global_named_dim_info:
    def __init__(self, named_dim_set: Set[str]) -> None:
        self.named_dim_set = named_dim_set

    def __enter__(self) -> None:
        for name in self.named_dim_set:
            _NAME_USE_CNT[name] = _NAME_USE_CNT.get(name, 0) + 1

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for name in self.named_dim_set:
            _NAME_USE_CNT[name] -= 1
            if _NAME_USE_CNT[name] == 0:
                _DIM_DICT.pop(name)
                _NAME_USE_CNT.pop(name)


def _check_item(expected_shape: ShapeDef,
                arg: Any,
                dim_dict: Optional[NamedDimMap] = None) -> _ShapeInfo:
    if expected_shape is not None:
        is_comp = is_compatible(arg.shape, expected_shape, dim_dict)
        return _ShapeInfo(is_comp, expected_shape, arg.shape)
    else:
        return _ShapeInfo(True)


def _collect_names(x: Optional[Union[str, int]], named_dim_set: Set[str]) -> None:
    if isinstance(x, str) and x != '...':
        named_dim_set.add(x)
