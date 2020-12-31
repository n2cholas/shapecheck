import functools
import inspect
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Union


class _ShapeInfo(NamedTuple):
    is_compatible: bool
    expected_shape: Optional[Sequence[Union[str, int]]] = None
    actual_shape: Optional[Sequence[int]] = None
    arg_name: Optional[str] = None


class ShapeError(RuntimeError):
    def __init__(self,
                 fn_name: str,
                 named_dims: Dict[str, int],
                 input_info: Iterable[_ShapeInfo],
                 output_info: Optional[_ShapeInfo] = None):
        strings = [
            f'in function {fn_name}.', f'Named Dimensions: {named_dims}.', 'Input:'
        ]
        for inp in input_info:
            if inp.expected_shape:
                info = (f'Argument: {inp.arg_name} '
                        f'Expected Shape: {inp.expected_shape} '
                        f'Actual Shape: {inp.actual_shape}.')
                if inp.is_compatible:
                    strings.append(_green_highlight(f'    Match:    {info}'))
                else:
                    strings.append(_red_highlight(f'    MisMatch: {info}'))
            else:
                strings.append(f'    Skipped:  Argument: {inp.arg_name}.')

        if output_info:
            strings.append('Output:')
            strings.append(
                _red_highlight(f'    MisMatch: Argument: {output_info.arg_name} '
                               f'Expected Shape: {output_info.expected_shape} '
                               f'Actual Shape: {output_info.actual_shape}.'))

        super().__init__('\n'.join(strings))


def is_compatible(shape: Sequence[int],
                  expected_shape: Sequence[Union[int, str]],
                  dim_dict: Optional[Dict[str, int]] = None) -> bool:
    if dim_dict is None:
        dim_dict = {}

    has_ellipsis = any(
        isinstance(s, str) and s.endswith('...')
        for s in expected_shape)  # type: ignore
    if not has_ellipsis and len(shape) != len(expected_shape):
        return False
    if len(shape) < len(expected_shape) - int(has_ellipsis):
        return False

    s_ind, es_ind = 0, 0
    while es_ind < len(expected_shape):
        if s_ind >= len(shape):
            exp_dim = expected_shape[es_ind]
            return isinstance(exp_dim, str) and exp_dim.endswith('...')  # type: ignore
        dim, exp_dim = shape[s_ind], expected_shape[es_ind]

        if isinstance(exp_dim, str):
            if exp_dim.endswith('...'):
                s_ind += len(shape) - len(expected_shape)
            elif exp_dim in dim_dict:
                exp_dim = dim_dict[exp_dim]
            else:
                dim_dict[exp_dim] = dim
                exp_dim = dim

        if (isinstance(exp_dim, int) and exp_dim != -1 and exp_dim is not None and
                exp_dim != dim):
            return False

        s_ind += 1
        es_ind += 1

    return True


def check_shapes(*in_shapes, out=None) -> Callable[[Callable], Callable]:
    in_shapes = [str_to_shape(in_s) if in_s else in_s
                 for in_s in in_shapes]  # type:ignore  # yapf: disable
    if out is not None:
        out = str_to_shape(out)

    def decorator(f: Callable) -> Callable:
        argspec = inspect.getfullargspec(f)
        all_args = argspec.args + argspec.kwonlyargs
        expected_shapes = list(zip(all_args, in_shapes))

        @functools.wraps(f)
        def inner(*args):
            assert len(args) == len(in_shapes)

            any_errors, dim_dict, input_info = False, {}, []
            for (arg_name, expected_shape), arg in zip(expected_shapes, args):
                if expected_shape is not None:
                    is_comp = is_compatible(arg.shape, expected_shape, dim_dict)
                    input_info.append(
                        _ShapeInfo(is_comp, expected_shape, arg.shape, arg_name))
                    any_errors |= not is_comp
                else:
                    input_info.append(_ShapeInfo(True, arg_name=arg_name))
            if any_errors:
                raise ShapeError(f.__name__, dim_dict, input_info)

            output = f(*args)

            if out is not None and not is_compatible(output.shape, out, dim_dict):
                raise ShapeError(f.__name__, dim_dict, input_info,
                                 _ShapeInfo(False, out, output.shape))

            return output

        return inner

    return decorator


def _green_highlight(s):
    return f'\x1b[6;30;42m{s}\x1b[0m'


def _red_highlight(s):
    return f'\x1b[6;30;41m{s}\x1b[0m'


def str_to_shape(string: str) -> Sequence[Union[int, str]]:
    shape: List[Union[int, str]] = []
    has_ellipsis = False
    for s in string.split(','):
        s = s.strip()
        if s.endswith('...'):
            if has_ellipsis:
                raise RuntimeError('Each shape can have at most one \'...\'.'
                                   f'Got {string}')
            has_ellipsis = True
            shape.append(s)
        else:
            try:
                shape.append(int(s))
            except ValueError:
                shape.append(s)
    return shape
