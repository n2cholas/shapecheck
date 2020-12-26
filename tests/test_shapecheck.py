import numpy as np
import pytest

from shapecheck import check_shape


def test_basic():
    @check_shape((3,), (4,), out_=(2,))
    def f(x, y):
        return x[:2]**2 + y[:2]**2

    f(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    with pytest.raises(AssertionError):
        f(np.array([1, 2, 3]), np.array([2, 3, 4]))


def test_named_dim():
    @check_shape((3, 'N'), ('N',), out_=(1, 'N'))
    def f(x, y):
        return (x + y).sum(0, keepdims=True)

    f(np.ones((3, 5)), np.ones((5,)))
    with pytest.raises(AssertionError):
        f(np.ones((3, 4)), np.ones((5,)))


def test_named_dim_one_arg():
    @check_shape(('A', 'A', 'N'), out_=('N',))
    def f(x):
        return x.sum((0, 1))

    f(np.ones((5, 5, 7)))
    with pytest.raises(AssertionError):
        f(np.ones((6, 5, 7)))


def test_any_dim():
    @check_shape(('N', -1), out_=('N', 1))
    def f(x):
        return x.sum(-1, keepdims=True)

    f(np.ones((5, 3)))
    f(np.ones((5, 7)))


def test_ndim_mismatch():
    @check_shape((-1, -1))
    def f(x):
        return x

    f(np.ones((1, 2)))
    with pytest.raises(AssertionError):
        f(np.ones((1,)))
    with pytest.raises(AssertionError):
        f(np.ones((1, 2, 3)))
