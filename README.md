
# ShapeCheck

![Build & Tests](https://github.com/n2cholas/shapecheck/workflows/Build%20and%20Tests/badge.svg)
[![codecov](https://codecov.io/gh/n2cholas/shapecheck/branch/main/graph/badge.svg?token=KAW5F029PM)](https://codecov.io/gh/n2cholas/shapecheck)

Framework-agnostic library for checking array/tensor shapes at runtime.

Finding the root of shape mismatches can be troublesome, especially with
broadcasting rules and mutable arrays. Documenting shapes with comments can
become stale as code evolves. This library aims to solve both of those problems
by ensuring function pre and post shape expectations are met. The concise
syntax for expressing shapes serves to document code as well, so new users can
quickly understand what's going on.

With frameworks like JAX or TensorFlow, "runtime" is actually "compile" or
"trace" time, so you don't pay any cost during execution. For frameworks like
PyTorch, asynchronous execution will hide the cost of shape checking. You only
pay a small overhead with synchronous, eager frameworks like numpy.

## Install Library

```bash
pip install --upgrade git+https://github.com/n2cholas/shapecheck.git
```

## Usage

```python
import numpy as np
from shapecheck import check_shapes

@check_shapes('-1,N', 'N', None, '3,N', out_='3,N')
def f(a, b, c, d):
    # a must be rank 2, where the first dim can be anything.
    # b must be rank 1, where the first dim must be N (a's second dim)
    # c will not be checked
    # d must be rank 2, where the first dim is 3, the second dim is N
    # since we specified `out_=`, the output shape will be checked as well
    return (a + b).sum(0, keepdims=True) + d

f(np.ones((7, 5)), np.ones(5), 'anything', np.ones((3, 5)))  # succeeds
f(np.ones((2, 6)), np.ones(6), np.ones(1), np.ones((3, 6)))  # succeeds
f(np.ones((2, 6)), np.ones(5), np.ones(1), np.ones((3, 6)))  # fails
```

The last statement throws a `ShapeError` with an informative message.

```
shapecheck.exception.ShapeError: in function f.
Named Dimensions: {'N': 6}.
Input:
    Match:    Argument: a Expected Shape: (-1, 'N') Actual Shape: (2, 6).
    MisMatch: Argument: b Expected Shape: ('N',) Actual Shape: (5,).
    Skipped:  Argument: c.
    Match:    Argument: d Expected Shape: (3, 'N') Actual Shape: (3, 6).
```

Above, the named dimensions have one letter names, but they can be strings of
any length.

If you have a function with shapechecking that calls many other functions
with shapechecking, you can optionally enforce that dimensions with the same
letter name in the parent correspond to the same sized dimension in the children.
That is, you can check that a function's input named dimensions match the same
named dimensions of all functions higher in the call stack. For example:

```python
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
parent_fn_2(np.ones(5), np.ones(6), np.ones(7))  # fails
```

Here, we swapped 'N' and 'M'. So, `parent_fn_1` succeeds because the inputs are
compatible for each individual function. But `parent_fn_2` fails because the
named dimensions are inconsistent between the parent and child functions. The
following error would be produced:

```
shapecheck.exception.ShapeError: in function child_fn.
Named Dimensions: {'M': 5, 'N': 6, 'R': 7, 'O': 7}.
Input:
    MisMatch: Argument: x Expected Shape: ('N',) Actual Shape: (5,).
    MisMatch: Argument: y Expected Shape: ('M',) Actual Shape: (6,).
    Match:    Argument: z Expected Shape: ('O',) Actual Shape: (7,).
```

This library also supports variadic dimensions. You can use '...' to indicate 0
or more dimensions:

```python
@check_shapes('1,...,1', '...,1,1')
def g(a, b):
    pass

g(np.ones((1, 3, 4, 1)), np.ones((2, 1, 1)))  # succeeds
g(np.ones((1, 1)), np.ones((1, 1)))  # succeeds
g(np.ones((2, 3, 4, 1)), np.ones((1, 1)))  # fails
```

The last statement fails with the following error:

```
shapecheck.exception.ShapeError: in function g.
Named Dimensions: {}.
Input:
    MisMatch: Argument: a Expected Shape: (1, '...', 1) Actual Shape: (2, 3, 4, 1).
    Match:    Argument: b Expected Shape: ('...', 1, 1) Actual Shape: (1, 1).
```

You can also name variadic dimensions, to ensure that a contiguous sequence of
dimensions match between arguments. For example:

```python
@check_shapes('batch,variadic...', 'variadic...')
def h(a, b):
    pass

h(np.ones((7, 1, 2)), np.ones((1, 2)))  # succeeds
h(np.ones((6, 2)), np.ones((1, 1)))  # fails
h(np.ones((6, 2)), np.ones((1)))  # fails
```

You can used nested lists/tuples/dictionaries as inputs, as demonstrated below:

```python
@check_shapes(('N,1', 'N'), '1,2', out_={'key1': ('N,1', 'N'), 'key2': ('1,2')})
def f(a, b):
    return {'key1': (a[1], a[1]), 'key2': b.sum()}

f((np.ones((7, 1)), np.ones((7,))), np.ones((1, 2)))  # fails
```

Which fails with the following error:

```
shapecheck.exception.ShapeError: in function f.
Named Dimensions: {'N': 7}.
Input:
    Argument: a  Type: <class 'tuple'>
        Match:    Ind: 0 Expected Shape: ('N', 1) Actual Shape: (7, 1).
        Match:    Ind: 1 Expected Shape: ('N',) Actual Shape: (7,).
    Match:    Argument: b Expected Shape: (1, 2) Actual Shape: (1, 2).
Output:  Type: <class 'dict'>
    Key: key1
        MisMatch: Ind: 0 Expected Shape: ('N', 1) Actual Shape: (7,).
        Match:    Ind: 1 Expected Shape: ('N',) Actual Shape: (7,).
    MisMatch: Key: key2 Expected Shape: (1, 2) Actual Shape: ().
```

You can enable/disable shapechecking globally as shown below:

```python
from shapecheck import is_checking_enabled, set_checking_enabled

assert is_checking_enabled()
set_checking_enabled(False)
assert not is_checking_enabled()
set_checking_enabled(True)
assert is_checking_enabled()
```

Or via a context manager:

```python
assert is_checking_enabled()
with set_checking_enabled(False):
    assert not is_checking_enabled()
assert is_checking_enabled()
```

## Run Tests

```bash
git clone https://github.com/n2cholas/shapecheck.git
cd shapecheck
pip install --upgrade pip setuptools
python setup.py develop
pip install -r requirements-dev.txt
pytest tests/ -vvv --cov shapecheck --cov-report term-missing
```

## Contributing

You can optionally use `pre-commit` to fix formatting and other issues.
`pre-commit` is in the `requirement-dev.txt`, so it should already be
installed. Set it up via:

```bash
pre-commit install
```

Then, whenever you commit:

```bash
git add .
git commit -m "Make change"
# files may be modified by the pre-commit hooks or may need modification
# fix the files that need to be fixed, then
git commit -am "Make change"  # adds modified files and commits
```

If you don't use pre-commit, there is a GitHub action to automatically
format your code when you push to `main`.

## Design Decisions

- Using strings instead of tuples to specify shapes.
  - Pros:
    - more concise, fewer paranthesis makes it more human readable.
    - can use more expressive syntax, e.g. named variadic dims (WIP).
  - Cons:
    - need string validation at runtime
    - more prone to errors.
- Decorator instead of type hints.
  - Pros:
    - this library is for runtime checking, not static analysis
    - type hints would interfere with existing type hints and static
      analyzers
  - Cons:
    - more verbose, adds visual noise
    - if you change argument order or refactor, you need to remember
      to make the same changes to the decorator, which is error prone.
