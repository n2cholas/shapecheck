from typing import Dict, Iterator, NamedTuple, Optional, Tuple

from .utils import NamedDimMap, NestedStruct, ShapeDef


class _ShapeInfo(NamedTuple):
    is_compatible: bool
    expected_shape: Optional[ShapeDef] = None
    actual_shape: Optional[Tuple[int, ...]] = None

    def __str__(self) -> str:
        if self.expected_shape:
            info = ('{} '
                    f'Expected Shape: {self.expected_shape} '
                    f'Actual Shape: {self.actual_shape}.')
            if self.is_compatible:
                return _style_text(f'Match:    {info}', _styles.GREEN)
            else:
                return _style_text(f'MisMatch: {info}', _styles.RED)
        else:
            return 'Skipped:  {}.'


class ShapeError(RuntimeError):
    """Error when a given array shape does not match the expcted shape."""
    def __init__(self,
                 fn_name: str,
                 named_dims: NamedDimMap,
                 input_info: Dict[str, _ShapeInfo],
                 output_info: Optional[_ShapeInfo] = None) -> None:  # noqa: D107
        strings = [
            f'in function {fn_name}.', f'Named Dimensions: {named_dims}.', 'Input:'
        ]

        for arg_name, info in input_info.items():
            if isinstance(info, _ShapeInfo):
                strings.append('    ' + str(info).format(f'Argument: {arg_name}'))
            else:
                strings.append(f'    Argument: {arg_name}  Type: {type(info)}')
                strings.extend(_nested_shape_info_to_strs(info, indent=8))

        if output_info:
            if isinstance(output_info, _ShapeInfo):
                strings.append('Output:')
                strings.append('    ' + str(output_info).format(''))
            else:
                strings.append(f'Output:  Type: {type(output_info)}')
                strings.extend(_nested_shape_info_to_strs(output_info, indent=4))

        super().__init__('\n'.join(strings))


class _styles:
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    HEADER = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'


def _style_text(string: str, style: str):
    return f'{style}{string}\033[0m'


def _nested_shape_info_to_strs(info: NestedStruct[_ShapeInfo],
                               indent: int) -> Iterator[str]:
    if isinstance(info, dict):
        info_gen = ((f'Key: {k}', v) for k, v in info.items())
    elif isinstance(info, (list, tuple)):
        info_gen = ((f'Ind: {k}', v) for k, v in enumerate(info))

    for k, v in info_gen:
        if isinstance(v, _ShapeInfo):
            yield indent * ' ' + str(v).format(k)
        else:
            yield indent * ' ' + k
            yield from _nested_shape_info_to_strs(v, indent=indent + 4)
