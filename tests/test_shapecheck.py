import numpy as np
import pytest

from shapecheck import ShapeError, check_shape

from .utils import CaptureStdOut


def test_basic():
    @check_shape('3', '4', out='2')
    def f(x, y):
        return x[:2]**2 + y[:2]**2

    f(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    with pytest.raises(ShapeError):
        f(np.array([1, 2, 3]), np.array([2, 3, 4]))


def test_named_dim():
    @check_shape('3,N', 'N', out='1,N')
    def f(x, y):
        return (x + y).sum(0, keepdims=True)

    f(np.ones((3, 5)), np.ones((5,)))
    with pytest.raises(ShapeError):
        f(np.ones((3, 4)), np.ones((5,)))


def test_named_dim_one_arg():
    @check_shape('A,A,N', out='N')
    def f(x):
        return x.sum((0, 1))

    f(np.ones((5, 5, 7)))
    with pytest.raises(ShapeError):
        f(np.ones((6, 5, 7)))


def test_any_dim():
    @check_shape('N,-1', out='N,1')
    def f(x):
        return x.sum(-1, keepdims=True)

    f(np.ones((5, 3)))
    f(np.ones((5, 7)))


def test_ndim_mismatch():
    @check_shape('-1,-1')
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

        @check_shape('3,A,A,N', out='N')
        def f(x):
            return x.sum((0, 2, 1))

        f(np.ones((3, 5, 5, 7)))
        with pytest.raises(ShapeError):
            f(np.ones((3, 6, 5, 7)))

    assert len(output) == 0


def test_readme_example():
    import numpy as np

    from shapecheck import check_shape

    @check_shape('-1,N', 'N', None, '3,N', out='3,N')
    def f(a, b, c, d):
        return (a + b).sum(0, keepdims=True) + d

    f(np.ones((7, 5)), np.ones(5), 'anything', np.ones((3, 5)))  # succeeds
    f(np.ones((2, 6)), np.ones(6), np.ones(1), np.ones((3, 6)))  # succeeds
    with pytest.raises(ShapeError):
        f(np.ones((2, 6)), np.ones(5), np.ones(1), np.ones((3, 6)))  # fails


def test_non_array_args():
    @check_shape(None, '2,N', None)
    def f(x, y, z):
        return 1

    f('some string', np.ones((2, 5)), np.ones((5,)))
    f(np.ones((1, 2, 3)), np.ones((2, 6)), 'non-array object')
    with pytest.raises(ShapeError):
        f(np.ones((1, 1)), np.ones((3, 5)), np.ones((5,)))
    with pytest.raises(ShapeError):
        f('another-test', np.ones((3, 6)), 'non-array object')
