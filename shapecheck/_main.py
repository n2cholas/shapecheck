def is_compatible(shape, expected_shape):
    return shape == expected_shape

def check_shape(*in_shapes, out_=None):
    def decorator(f):
        def inner(*args):
            for in_shape, arg in zip(in_shapes, args):
                assert is_compatible(arg.shape, in_shape)
            out = f(*args)
            if out_: assert is_compatible(out.shape, out_)
            return out
        return inner
    return decorator