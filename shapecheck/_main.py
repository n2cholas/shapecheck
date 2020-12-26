from functools import wraps


def is_compatible(shape, expected_shape, dim_dict=None):
    if dim_dict is None:
        dim_dict = {}
    if len(shape) != len(expected_shape):
        return False
    for dim, exp_dim in zip(shape, expected_shape):
        if isinstance(exp_dim, str):
            if exp_dim in dim_dict:
                exp_dim = dim_dict[exp_dim]
            else:
                dim_dict[exp_dim] = dim
                exp_dim = dim

        if exp_dim != -1 and exp_dim is not None and exp_dim != dim:
            return False
    return True


def check_shape(*in_shapes, out=None):
    in_shapes = [_str_to_shape(in_s) for in_s in in_shapes]
    if out is not None:
        out = _str_to_shape(out)

    def decorator(f):
        @wraps(f)
        def inner(*args):
            assert len(args) == len(in_shapes)
            dim_dict = {}
            for in_shape, arg in zip(in_shapes, args):
                assert is_compatible(arg.shape, in_shape, dim_dict=dim_dict)
            output = f(*args)
            if out:
                assert is_compatible(output.shape, out, dim_dict=dim_dict)
            return output

        return inner

    return decorator


def _try_int(s):
    # s.isnumeric() fails with -1
    try:
        return int(s)
    except ValueError:
        return s


def _str_to_shape(string):
    return tuple(_try_int(s.strip()) for s in string.split(','))
