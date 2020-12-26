import numpy as np
import pytest

from shapecheck import check_shape


def test_basic():
    @check_shape((3,), (4,), out_=(2,))
    def func(x, y):
        return x[:2]**2 + y[:2]**2

    func(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    with pytest.raises(AssertionError):
        func(np.array([1, 2, 3]), np.array([2, 3, 4]))


def test_named_dim():
    @check_shape(('N', 3), ('N',), out_=('N', 1))
    def func(x, y):
        return (x + y).sum(-1, keepdims=True)

    func(np.ones((5, 3)), np.ones((5, 1)))
    with pytest.raises(AssertionError):
        func(np.ones((4, 3)), np.ones((5, 1)))


def test_any_dim():
    @check_shape(('N', -1), out_=('N', 1))
    def func(x):
        return x.sum(-1, keepdims=True)

    func(np.ones((5, 3)))
    func(np.ones((5, 7)))
