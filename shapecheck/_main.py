from functools import wraps


def is_compatible(shape, expected_shape, dim_dict=None):
    if dim_dict is None:
        dim_dict = {}
    for dim, exp_dim in zip(shape, expected_shape):
        if isinstance(exp_dim, str):
            if exp_dim in dim_dict:
                print('Found')
                exp_dim = dim_dict[exp_dim]
            else:
                print('Adding ', exp_dim, dim, shape, expected_shape)
                dim_dict[exp_dim] = dim
                exp_dim = dim

        if exp_dim != -1 and exp_dim is not None and exp_dim != dim:
            return False
    return True


def check_shape(*in_shapes, out_=None):
    def decorator(f):
        @wraps(f)
        def inner(*args):
            dim_dict = {}
            for in_shape, arg in zip(in_shapes, args):
                assert is_compatible(arg.shape, in_shape, dim_dict=dim_dict)
            out = f(*args)
            if out_:
                assert is_compatible(out.shape, out_, dim_dict=dim_dict)
            return out

        return inner

    return decorator
