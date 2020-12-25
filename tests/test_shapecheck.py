import numpy as np

from shapecheck import check_shape


def test_basic():
    @check_shape((3,), (4,), out_=(2,))
    def func(x, y):
        return x[:2]**2 + y[:2]**2

    func(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
