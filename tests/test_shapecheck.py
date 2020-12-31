import numpy as np
import pytest

from shapecheck import ShapeError, check_shapes, is_compatible, str_to_shape

from .utils import CaptureStdOut


def test_basic():
    @check_shapes('3', '4', out='2')
    def f(x, y):
        return x[:2]**2 + y[:2]**2

    f(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    with pytest.raises(ShapeError):
        f(np.array([1, 2, 3]), np.array([2, 3, 4]))


def test_named_dim():
    @check_shapes('3,N', 'N', out='1,N')
    def f(x, y):
        return (x + y).sum(0, keepdims=True)

    f(np.ones((3, 5)), np.ones((5,)))
    with pytest.raises(ShapeError):
        f(np.ones((3, 4)), np.ones((5,)))


def test_named_dim_one_arg():
    @check_shapes('A,A,N', out='N')
    def f(x):
        return x.sum((0, 1))

    f(np.ones((5, 5, 7)))
    with pytest.raises(ShapeError):
        f(np.ones((6, 5, 7)))


def test_any_dim():
    @check_shapes('N,-1', out='N,1')
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

        @check_shapes('3,A,A,N', out='N')
        def f(x):
            return x.sum((0, 2, 1))

        f(np.ones((3, 5, 5, 7)))
        with pytest.raises(ShapeError):
            f(np.ones((3, 6, 5, 7)))

    assert len(output) == 0


def test_readme_example():
    import numpy as np

    from shapecheck import check_shapes

    @check_shapes('-1,N', 'N', None, '3,N', out='3,N')
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


@pytest.mark.parametrize('string', ['...,...,...', 'a,...,b,...', '...,1,...'])
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
