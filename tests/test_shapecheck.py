import numpy as np
import pytest

from shapecheck import (ShapeError, check_shapes, is_checking_enabled, is_compatible,
                        set_checking_enabled, str_to_shape)

from .utils import CaptureStdOut


def test_basic():
    @check_shapes('3', '4', out_='2')
    def f(x, y):
        return x[:2]**2 + y[:2]**2

    f(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    with pytest.raises(ShapeError):
        f(np.array([1, 2, 3]), np.array([2, 3, 4]))


def test_named_dim():
    @check_shapes('3,N', 'N', out_='1,N')
    def f(x, y):
        return (x + y).sum(0, keepdims=True)

    f(np.ones((3, 5)), np.ones((5,)))
    with pytest.raises(ShapeError):
        f(np.ones((3, 4)), np.ones((5,)))


def test_named_dim_one_arg():
    @check_shapes('A,A,N', out_='N')
    def f(x):
        return x.sum((0, 1))

    f(np.ones((5, 5, 7)))
    with pytest.raises(ShapeError):
        f(np.ones((6, 5, 7)))


def test_any_dim():
    @check_shapes('N,-1', out_='N,1')
    def f(x):
        return x.sum(-1, keepdims=True)

    f(np.ones((5, 3)))
    f(np.ones((5, 7)))


def test_ndim_mismatch():
    @check_shapes('-1,-1')
    def f(x):
        return x

    f(np.ones((1, 2)))
    with pytest.raises(ShapeError):
        f(np.ones((1,)))
    with pytest.raises(ShapeError):
        f(np.ones((1, 2, 3)))


def test_no_stdout():
    # Prevent pushing debug messages.
    with CaptureStdOut() as output:

        @check_shapes('3,A,A,N', out_='N')
        def f(x):
            return x.sum((0, 2, 1))

        f(np.ones((3, 5, 5, 7)))
        with pytest.raises(ShapeError):
            f(np.ones((3, 6, 5, 7)))

    assert len(output) == 0


def test_readme_example():
    import numpy as np

    from shapecheck import check_shapes

    @check_shapes('-1,N', 'N', None, '3,N', out_='3,N')
    def f(a, b, c, d):
        return (a + b).sum(0, keepdims=True) + d

    f(np.ones((7, 5)), np.ones(5), 'anything', np.ones((3, 5)))  # succeeds
    f(np.ones((2, 6)), np.ones(6), np.ones(1), np.ones((3, 6)))  # succeeds
    with pytest.raises(ShapeError):
        f(np.ones((2, 6)), np.ones(5), np.ones(1), np.ones((3, 6)))  # fails

    @check_shapes('1,...,1', '...,1,1')
    def g(a, b):
        pass

    g(np.ones((1, 3, 4, 1)), np.ones((2, 1, 1)))  # succeeds
    g(np.ones((1, 1)), np.ones((1, 1)))  # succeeds
    with pytest.raises(ShapeError):
        g(np.ones((2, 3, 4, 1)), np.ones((1, 1)))  # fails

    @check_shapes('batch,variadic...', 'variadic...')
    def h(a, b):
        pass

    h(np.ones((7, 1, 2)), np.ones((1, 2)))  # succeeds
    with pytest.raises(ShapeError):
        h(np.ones((6, 2)), np.ones((1, 1)))  # fails
    with pytest.raises(ShapeError):
        h(np.ones((6, 2)), np.ones((1)))  # fails


def test_non_array_args():
    @check_shapes(None, '2,N', None)
    def f(x, y, z):
        return 1

    f('some string', np.ones((2, 5)), np.ones((5,)))
    f(np.ones((1, 2, 3)), np.ones((2, 6)), 'non-array object')
    with pytest.raises(ShapeError):
        f(np.ones((1, 1)), np.ones((3, 5)), np.ones((5,)))
    with pytest.raises(ShapeError):
        f('another-test', np.ones((3, 6)), 'non-array object')


@pytest.mark.parametrize('string, shape', [('N,1,3,M', ('N', 1, 3, 'M')),
                                           ('N, 1, 3, M', ('N', 1, 3, 'M')),
                                           ('...,a,1', ('...', 'a', 1)),
                                           ('1, ... ,2', (1, '...', 2)),
                                           ('a,b,c,...', ('a', 'b', 'c', '...')),
                                           ('...', ('...',))])
def test_shape_to_str(string, shape):
    result = str_to_shape(string)
    for a, b in zip(shape, result):
        assert a == b, f'Expected: {shape} Got: {result}'


@pytest.mark.parametrize('string', [
    '...,...,...', 'a,...,b,...', '...,1,...', (1, 2), 3, 4.0, [5.0], ['1,2'], ('1,2',)
])
def test_shape_to_str_error(string):
    with pytest.raises(RuntimeError):
        str_to_shape(string)


@pytest.mark.parametrize('shape, expected_shape', [
    ((3, 2, 3), ('n', 2, 'n')),
    ((3, 2, 3), ('n', '...', 2, 'n')),
    ((3, 1, 1, 2, 3), ('n', '...', 2, 'n')),
    ((3, 2, 3), ('...', 'n', 2, 'n')),
    ((1, 1, 3, 2, 3), ('...', 'n', 2, 'n')),
    ((3, 2, 3), ('n', 2, 'n', '...')),
    ((3, 2, 3, 1, 1), ('n', 2, 'n', '...')),
    ((3, 2, 3), ('...',)),
])
def test_compatible_variadic_shapes(shape, expected_shape):
    assert is_compatible(shape, expected_shape)


@pytest.mark.parametrize('shape, expected_shape', [
    ((3, 3, 3), ('n', 2, 'n')),
    ((3, 2, 4), ('n', '...', 2, 'n')),
    ((3, 1, 1, 3, 3), ('n', '...', 2, 'n')),
    ((4, 2, 3), ('...', 'n', 2, 'n')),
    ((1, 1, 2, 3), ('...', 'n', 2, 'n')),
    ((3, 3), ('n', 2, 'n', '...')),
    ((2, 3, 1, 1), ('n', 2, 'n', '...')),
])
def test_incompatible_variadic_shapes(shape, expected_shape):
    assert not is_compatible(shape, expected_shape)


@pytest.mark.parametrize('e_shape1, e_shape2, shape1, shape2', [
    (('n...,1,1', 'n...', (1, 2, 3, 1, 1), (1, 2, 3))),
    (('...,1,1', 'n...', (1, 2, 3, 1, 1), (1, 2, 3))),
    (('n...,2,2', '1,n...', (2, 2), (1,))),
    (('n...,1,1', 'a...', (1, 2, 3, 1, 1), (1, 3))),
    (('1,2,a...,3,4', '6,a...,7', (1, 2, 9, 9, 3, 4), (6, 9, 9, 7))),
    (('1,2,a...,3,4', '6,a...,7', (1, 2, 9, 3, 4), (6, 9, 7))),
    (('1,2,a...,3,4', '6,a...,7', (1, 2, 3, 4), (6, 7))),
])
def test_named_variadic_shapes(e_shape1, e_shape2, shape1, shape2):
    @check_shapes(e_shape1, e_shape2)
    def f(a, b):
        pass

    f(np.ones(shape1), np.ones(shape2))


@pytest.mark.parametrize('e_shape1, e_shape2, shape1, shape2', [
    (('n...,1,1', 'n...', (1, 2, 3, 1, 1), (1, 3, 3))),
    (('n...,1,1', 'n...', (1, 2, 3, 1, 1), (1, 3))),
    (('n...,2,2', '1,n...', (2, 2), (1, 1))),
    (('n...,2,2', 'n...', (2, 2), (1,))),
    (('n...,', 'n...', (2, 2), (1,))),
    (('1,2,a...,3,4', '6,a...,7', (1, 2, 8, 9, 3, 4), (6, 9, 9, 7))),
    (('1,2,a...,3,4', '6,a...,7', (1, 2, 7, 3, 4), (6, 9, 7))),
    (('1,2,a...,3,4', '6,a...,7', (1, 2, 3, 4), (6, 1, 7))),
])
def test_bad_named_variadic_shapes(e_shape1, e_shape2, shape1, shape2):
    @check_shapes(e_shape1, e_shape2)
    def f(a, b):
        pass

    with pytest.raises(ShapeError):
        f(np.ones(shape1), np.ones(shape2))


def test_incompatible_output():
    @check_shapes(out_='1,1')
    def f():
        return np.ones((1,))

    with pytest.raises(ShapeError):
        f()


def test_nested_structs():
    @check_shapes(('N,1', 'N'), '1,2', out_={'one': ('N,1', 'N'), 'two': ('1,2')})
    def f(one, two):
        return {'one': one, 'two': two}

    f((np.ones((7, 1)), np.ones((7,))), np.ones((1, 2)))
    with pytest.raises(ShapeError):
        f((np.ones((7, 1)), np.ones((6,))), np.ones((1, 2)))


def test_readme_nested_example():
    @check_shapes(('N,1', 'N'), '1,2', out_={'one': ('N,1', 'N'), 'two': ('1,2')})
    def f(one, two):
        return {'one': (one[1], one[1]), 'two': two.sum()}

    with pytest.raises(ShapeError):
        f((np.ones((7, 1)), np.ones((7,))), np.ones((1, 2)))


def test_readme_set_checking_enabled():
    from shapecheck import is_checking_enabled, set_checking_enabled

    assert is_checking_enabled()
    set_checking_enabled(False)
    assert not is_checking_enabled()
    set_checking_enabled(True)
    assert is_checking_enabled()
    with set_checking_enabled(False):
        assert not is_checking_enabled()
    assert is_checking_enabled()


def test_set_checking_enabled():
    @check_shapes('3', '4', out_='2')
    def f(x, y):
        return x[:2]**2 + y[:2]**2

    set_checking_enabled(False)
    assert not is_checking_enabled()
    f(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    f(np.array([1, 2, 3]), np.array([2, 3, 4]))

    @check_shapes('3', '4', out_='2')
    def g(x, y):
        return x[:2]**2 + y[:2]**2

    set_checking_enabled(True)
    assert is_checking_enabled()

    with pytest.raises(ShapeError):
        f(np.array([1, 2, 3]), np.array([2, 3, 4]))
    with pytest.raises(ShapeError):
        g(np.array([1, 2, 3]), np.array([2, 3, 4]))


def test_set_checking_enabled_context():
    @check_shapes('3', '4', out_='2')
    def f(x, y):
        return x[:2]**2 + y[:2]**2

    assert is_checking_enabled()
    with set_checking_enabled(False):
        assert not is_checking_enabled()
        f(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
        f(np.array([1, 2, 3]), np.array([2, 3, 4]))

        @check_shapes('3', '4', out_='2')
        def g(x, y):
            return x[:2]**2 + y[:2]**2

    assert is_checking_enabled()
    with pytest.raises(ShapeError):
        f(np.array([1, 2, 3]), np.array([2, 3, 4]))
    with pytest.raises(ShapeError):
        g(np.array([1, 2, 3]), np.array([2, 3, 4]))


def test_match_callees():
    @check_shapes('N', 'M', 'O', out_='N')
    def f(x, y, z):
        return x

    @check_shapes('N', 'M', 'R', match_callees_=True)
    def g(x, y, z):
        return f(x, y, z)

    g(np.ones(5), np.ones(6), np.ones(7))


def test_match_callees_error():
    @check_shapes('N', 'M', 'O', out_='N')
    def f(x, y, z):
        return x

    @check_shapes('M', 'N', 'R', match_callees_=True)
    def g(x, y, z):
        return f(x, y, z)

    with pytest.raises(ShapeError):
        g(np.ones(5), np.ones(6), np.ones(7))


def test_match_callees_complex():
    @check_shapes('a, v...', 'v...', out_='v...')
    def f(x, y):
        return x.sum(0) + y

    @check_shapes('v...')
    def g(x):
        return x.sum()

    @check_shapes('a', match_callees_=True)
    def h(x):
        a = np.ones((x.shape[0], 2, 3, 4))
        b = np.ones((2, 3, 4))
        f(a, b)
        return g(np.ones((5, 4, 3)))

    h(np.ones((8)))

    @check_shapes('a', match_callees_=True)
    def h(x):
        a = np.ones((x.shape[0] - 1, 2, 3, 4))
        b = np.ones((2, 3, 4))
        f(a, b)
        return g(np.ones((5, 4, 3)))

    with pytest.raises(ShapeError):
        h(np.ones((8)))


def test_match_callees_readme():
    @check_shapes('N', 'M', 'O', out_='N')
    def child_fn(x, y, z):
        return x

    @check_shapes('M', 'N', 'R')
    def parent_fn_1(x, y, z):
        return child_fn(x, y, z)

    @check_shapes('M', 'N', 'R', match_callees_=True)
    def parent_fn_2(x, y, z):
        return child_fn(x, y, z)

    parent_fn_1(np.ones(5), np.ones(6), np.ones(7))  # succeeds
    with pytest.raises(ShapeError):
        parent_fn_2(np.ones(5), np.ones(6), np.ones(7))  # fail


@pytest.mark.parametrize('cs_args, cs_kwargs, f_args, f_kwargs', [
    (('N', 'M', 'O', 'P'), {}, (1, 2, 3), {}),
    (('N', 'M', 'O', 'P'), {}, (1, 2), {'c': 3}),
    (('N', 'M', 'O',), {}, (1, 2, 3), {}),
    (('N', 'M'), {}, (1, 2), {'c': 3}),
    (('N', 'M', 'O', 'P'), {}, (1,), {'c': 3, 'b': 2}),
    (('N', 'M', 'O'), {'d': 'P'}, (1, 2, 3), {}),
    (('N', 'M'), {'c': 'O', 'd': 'P'}, (1, 2, 3), {}),
    (('N',), {'b': 'M', 'c': 'O', 'd': 'P'}, (1, 2), {'c': 3}),
    ((), {'a': 'N', 'b': 'M', 'c': 'O', 'd': 'P'}, (1, 2, 3), {}),
    ((), {'a': 'N', 'b': 'M', 'c': 'O', 'd': 'P'}, (1, 2), {'c': 3}),
])  # yapf: disable
def test_check_shapes_signature(cs_args, cs_kwargs, f_args, f_kwargs):
    # TODO: write more rigorous shape signature tests
    @check_shapes(*cs_args, **cs_kwargs)
    def f(a, b, c, *, d):
        pass

    f_kwargs = {k: np.ones(v) for k, v in f_kwargs.items()}
    f(*map(np.ones, f_args), d=np.ones(4), **f_kwargs)


def test_readme_example2():
    import numpy as np

    from shapecheck import check_shapes

    @check_shapes({'imgs': 'N,W,W,-1', 'labels': 'N,1'}, aux_info=None, out_='N')
    def per_eg_loss(batch, aux_info):
        # do something with aux_info
        return (batch['imgs'].mean((1, 2, 3)) - batch['labels'].squeeze())**2

    per_eg_loss({'imgs': np.ones((3, 2, 2, 1)), 'labels': np.ones((3, 1))}, np.ones(1))
    per_eg_loss({'imgs': np.ones((5, 3, 3, 4)), 'labels': np.ones((5, 1))}, 'any')
    with pytest.raises(ShapeError):
        per_eg_loss({'imgs': np.ones((3, 5, 2, 1)), 'labels': np.ones((3, 1))}, 'any')
