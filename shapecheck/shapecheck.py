import functools
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, cast

from .exception import ShapeError, _ShapeInfo
from .utils import NamedDimMap, NestedStruct, ShapeDef, iterate_nested, map_nested

__all__ = [
    'check_shapes', 'is_compatible', 'str_to_shape', 'set_checking_enabled',
    'is_checking_enabled'
]

# TODO: make these thread local
_CHECKING_ENABLED = True
_MATCH_CALLEES = False
_DIM_DICT: NamedDimMap = {}
_NAME_USE_CNT: Dict[str, int] = {}


def is_compatible(shape: Tuple[int],
                  expected_shape: ShapeDef,
                  dim_dict: Optional[NamedDimMap] = None) -> bool:
    """Check whether shape is compatible with expected_shape.

    Args:
        shape: dimensions to be checked
        expected_shape: pattern shape to match against shape, which can consist
            of integers, named dimensions, and named variadic dimensions.
        dim_dict: dictionary mapping named dimensions/variadic dimensions to
            their concrete quantities (integers or tuples of integers).

    Returns:
        True if shape matches expected_shape, otherwise False.
    """
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
    """Convert string describing shape definition to a shape definition.

    Args:
        string: string with dimensions, named dimensions (e.g. N), named
            variadic dimensions (e.g. V...), or variadic dimensions (...)
            seperated by commas (spaces between items allowed).  e.g. 'batch,
            named_variadic...', '...,1,2,3', or 'N, M'.

    Returns:
        Shape definition corresponding to string, which is a tuple of integers
        or strings. If input string is None, returns None.
    """
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
                try:  # need try-catch because str.isnumeric() misses -1
                    yield int(s)
                except ValueError:
                    yield s

    return ShapeDef(gen()) if string else None


def check_shapes(
        *in_args: NestedStruct[Optional[str]],
        out_: Optional[NestedStruct[Optional[str]]] = None,
        match_callees_: bool = False,
        **in_kws: NestedStruct[Optional[str]]) -> Callable[[Callable], Callable]:
    """Return decorator that checks input/output shapes of decorated function.

    Specify the expected shapes in the same order or using the same parameter
    names as the decorated function. Unspecified arguments or arguments
    explicitly given an expected shape of None will not be checked.

    Args:
        *in_args: nested dict/list/tuples of strings describing the allowed
            shapes for the decorated function. the nesting structure should match
            the structure of the function inputs.
        out_: nested dict/list/tuples of strings describing allowed shape for
            output of the decorated function.
        match_callees_: whether or not the named dimensions/variadic dimensions
            of functions called by the decorated functions should match the named
            dimensions of the decorated function.
        **in_kws: same as *in_args but key word arguments.

    Returns:
        Decorator that will check input/output shapes of decorated function.
    """
    in_args, in_kws = map_nested(str_to_shape, (in_args, in_kws))
    out_ = map_nested(str_to_shape, out_)

    def decorator(f: Callable) -> Callable:
        arglist = cast(Sequence[str], inspect.signature(f).parameters.keys())
        expected_shapes = _params_to_named_params(in_args, in_kws, arglist)
        named_dim_set: Set[str] = set()

        @functools.wraps(f)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if not _CHECKING_ENABLED:
                return f(*args)

            named_args = _params_to_named_params(args, kwargs, arglist)

            if (match_callees_ or _MATCH_CALLEES) and not named_dim_set:
                for dim in iterate_nested((expected_shapes, out_)):
                    if isinstance(dim, str) and dim != '...':
                        named_dim_set.add(dim)

            with _update_global_named_dim_info(named_dim_set):
                dim_dict: NamedDimMap = _DIM_DICT if match_callees_ or _DIM_DICT else {}
                check_fn = functools.partial(_check_item, dim_dict=dim_dict)

                input_info = map_nested(check_fn, expected_shapes, named_args,
                                        stop_type=ShapeDef)  # yapf: disable
                input_info = cast(Dict[str, NestedStruct[_ShapeInfo]], input_info)
                if not all(s.is_compatible
                           for s in iterate_nested(input_info, stop_type=_ShapeInfo)):
                    raise ShapeError(f.__name__, dim_dict, input_info)

                with _match_callees_enabled(match_callees_ or _MATCH_CALLEES):
                    output = f(*args, **kwargs)

                output_info = map_nested(check_fn, out_, output, stop_type=ShapeDef)
                if not all(s.is_compatible
                           for s in iterate_nested(output_info, stop_type=_ShapeInfo)):
                    raise ShapeError(f.__name__, dim_dict, input_info, output_info)

            return output

        return inner

    return decorator


class set_checking_enabled:
    """Context manager to toggle shape checking on or off."""
    def __init__(self, mode: bool) -> None:  # noqa: D107
        global _CHECKING_ENABLED
        self.prev = _CHECKING_ENABLED
        _CHECKING_ENABLED = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _CHECKING_ENABLED
        _CHECKING_ENABLED = self.prev


def is_checking_enabled() -> bool:
    """Return whether shape checking is enabled."""
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


def _params_to_named_params(args: Tuple[Any, ...], kwargs: Dict[str, Any],
                            arglist: Sequence[str]) -> Dict[str, Any]:
    assert len(args) <= len(arglist)
    d = dict(zip(arglist, args))
    d.update(kwargs)
    for k in arglist:
        if k not in d:
            d[k] = None
    return d
