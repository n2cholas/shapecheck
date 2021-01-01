from typing import List, NamedTuple

from shapecheck.utils import map_nested, reduce_nested


class DummyNamedTuple(NamedTuple):
    a: int
    b: List[int]


def test_map_nested():
    struct = {'test1': DummyNamedTuple(1, [1, 2, 3]), 'test2': 1, 13: ((6, 7), (7,))}

    def f(x):
        return 2 * x + 1

    mapped_struct = map_nested(f, struct)
    assert mapped_struct == {
        'test1': DummyNamedTuple(f(1), [f(1), f(2), f(3)]),
        'test2': f(1),
        13: ((f(6), f(7)), (f(7),))
    }


def test_multi_map_nested():
    struct1 = [
        {
            'x': (1, 2),
            11: 3
        },
        DummyNamedTuple(5, [6, 7, 8]),
        {9, 10},
        11,
    ]
    struct2 = [
        {
            'x': (12, 13),
            11: 14
        },
        DummyNamedTuple(16, [17, 18, 19]),
        {20, 21},
        22,
    ]
    struct3 = [
        {
            'x': (14, 15),
            11: 16
        },
        DummyNamedTuple(18, [19, 20, 21]),
        {22, 23},
        24,
    ]

    def f(a, b, c):
        return a + b * c

    mapped_struct = map_nested(f, struct1, struct2, struct3)
    assert mapped_struct == [
        {
            'x': (1 + 12 * 14, 2 + 13 * 15),
            11: 3 + 14 * 16
        },
        DummyNamedTuple(5 + 16 * 18, [6 + 17 * 19, 7 + 18 * 20, 8 + 19 * 21]),
        {9 + 20 * 22, 10 + 21 * 23},
        11 + 22 * 24,
    ]


def test_map_nested_no_struct():
    def f(x):
        return 2 * x + 1

    struct = 13

    mapped_struct = map_nested(f, struct)
    assert mapped_struct == f(13)


def test_map_nested_stop_type():
    struct = [
        DummyNamedTuple(1, [1, 2, 3]),
        DummyNamedTuple(3, [5, 6, 7]),
    ]

    def f(x):
        return sum(x[1])

    mapped_struct = map_nested(f, struct, stop_type=DummyNamedTuple)
    assert mapped_struct == [sum([1, 2, 3]), sum([5, 6, 7])]


def test_reduce_nested():
    struct = {'test1': DummyNamedTuple(1, [1, 2, 3]), 'test2': 1, 13: ({6, 7}, (7,))}
    assert reduce_nested(lambda x, y: x + y, struct) == 28


def test_reduce_nested_no_struct():
    assert reduce_nested(lambda x, y: x + y, 3) == 3


def test_reduce_nested_initial():
    assert reduce_nested(lambda x, y: x and y, [], True)
    assert reduce_nested(lambda x, y: x and y, {}, True)
    assert reduce_nested(lambda x, y: x and y, tuple(), True)
    assert reduce_nested(lambda x, y: x and y, set(), True)


def test_reduce_nested_stop_type():
    class StopList(list):
        pass

    struct = [
        {
            'x': StopList([3, 3]),
            11: StopList([2])
        },
        (StopList([[2]]), StopList([3, 3, 3])),
        StopList([1]),
    ]
    reduced = reduce_nested(lambda x, y: x + y, struct, stop_type=StopList)
    assert reduced == StopList([3, 3, 2, [2], 3, 3, 3, 1])
