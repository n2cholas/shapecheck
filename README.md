
# ShapeCheck

![Build & Tests](https://github.com/n2cholas/shapecheck/workflows/Build%20and%20Tests/badge.svg)
[![codecov](https://codecov.io/gh/n2cholas/shapecheck/branch/main/graph/badge.svg?token=KAW5F029PM)](https://codecov.io/gh/n2cholas/shapecheck)

Framework-agnostic library for checking array/tensor shapes at runtime.

Finding the root of shape mismatches can be troublesome, especially with
broadcasting rules and mutable arrays. Comments documenting shapes can easily
become out of date as code evolves. This library aims to solve both of those
problems by ensuring function input/output shape expectations are met. The
concise syntax for expressing shapes serves to document code as well, so new
users can quickly understand what's going on.

With frameworks like JAX or TensorFlow, "runtime" is actually "compile" or
"trace" time, so you don't pay any cost during execution. For frameworks like
PyTorch, asynchronous execution will hide the cost of shape checking. You only
pay a small overhead with synchronous, eager frameworks like numpy.

## Install Library

From PyPI:

```bash
pip install --upgrade shapecheck
```

To install the latest version:

```bash
pip install --upgrade git+https://github.com/n2cholas/shapecheck.git
```

## Usage

```python
import numpy as np
from shapecheck import check_shapes

@check_shapes({'imgs': 'N,W,W,-1', 'labels': 'N,1'}, aux_info=None, out_='')
def loss_fn(batch, aux_info):
    # do something with aux_info
    diff = (batch['imgs'].mean((1, 2, 3)) - batch['labels'].squeeze())
    return np.mean(diff**2)

loss_fn({'imgs': np.ones((3, 2, 2, 1)), 'labels': np.ones((3,1))}, np.ones(1))
loss_fn({'imgs': np.ones((5, 3, 3, 4)), 'labels': np.ones((5,1))}, 'any')
# Below line fails:
loss_fn({'imgs': np.ones((3, 5, 2, 1)), 'labels': np.ones((3,1))}, 'any')
```

Error message:

```
shapecheck.exception.ShapeError: in function loss_fn.
Named Dimensions: {'N': 3, 'W': 5}.
Input:
    Argument: batch  Type: <class 'dict'>
        MisMatch: Key: imgs Expected Shape: ('N', 'W', 'W', -1) Actual Shape: (3, 5, 2, 1).
        Match:    Key: labels Expected Shape: ('N', 1) Actual Shape: (3, 1).
    Skipped:  Argument: aux_info.
```

In the above example, we compute the loss with a batch of data, which is a
dictionary with images and labels. We specify that we want `N` square images
which can have any number of channels (indicated by the `-1`).  Inputs to
`check_shape` can be arbitrarily nested dicts/lists/tuples, as long as the
structure of the shape specification matches the structure of the inputs to the
decorated function.

We also specify that `aux_info` shouldn't be checked. Equivalently, we could've
excluded it from the definition:

```python
@check_shapes({'imgs': 'N,W,W,-1', 'labels': 'N,1'}, out_='')
```

or passed it as a positional argument.

```python
@check_shapes({'imgs': 'N,W,W,-1', 'labels': 'N,1'}, None, out_='')
```

Finally, we specify the output shape should be a scalar via `out_=''`. All
non-input shape arguments to `check_shape` have an underscore after them so
they don't conflict with the decorated function's arguments (for now, just
`out_` and `match_callees_`).

If you have a function with shape-checking that calls many other functions with
shape-checking, you can optionally enforce that dimensions with the same letter
name in the caller correspond to the same sized dimension in the callees via
`match_callees_=True`.  That is, you can check that a function's input named
dimensions match the same named dimensions of all checked functions higher in
the call stack. For example:

```python
@check_shapes('M', 'N', 'O', out_='M')
def callee(a, b, c):
    return a

@check_shapes('M', 'N', 'R')
def caller_fn_1(x, y, z):
    return callee(y, x, z)

@check_shapes('M', 'N', 'R', match_callees_=True)
def caller_fn_2(x, y, z):
    return callee(y, x, z)

caller_fn_1(np.ones(5), np.ones(6), np.ones(7))  # succeeds
caller_fn_2(np.ones(5), np.ones(6), np.ones(7))  # fails
```

Here, we (accidentally) swapped `x` and `y` when calling `callee`.
`caller_fn_1` succeeds because the inputs are compatible when considering the
named dimensions for `callee_fn` alone. But `caller_fn_2` fails because the
named dimensions are inconsistent between the caller and the callee. The
following error would be produced:

```
shapecheck.exception.ShapeError: in function callee.
Named Dimensions: {'M': 5, 'N': 6, 'R': 7, 'O': 7}.
Input:
    MisMatch: Argument: a Expected Shape: ('M',) Actual Shape: (6,).
    MisMatch: Argument: b Expected Shape: ('N',) Actual Shape: (5,).
    Match:    Argument: c Expected Shape: ('O',) Actual Shape: (7,).
```

This library also supports variadic dimensions. You can use '...' to indicate 0
or more dimensions:

```python
@check_shapes('dim, ..., 1', '..., dim, 1')
def g(a, b):
    pass

g(np.ones((2, 3, 4, 1)), np.ones((5, 2, 1)))  # succeeds
g(np.ones((3, 1)), np.ones((3, 1)))  # succeeds
g(np.ones((2, 3, 4, 1)), np.ones((1, 1)))  # fails
```

The last statement fails with the following error, since `dim` doesn't match:

```
shapecheck.exception.ShapeError: in function g.
Named Dimensions: {'dim': 2}.
Input:
    Match:    Argument: a Expected Shape: ('dim', '...', 1) Actual Shape: (2, 3, 4, 1).
    MisMatch: Argument: b Expected Shape: ('...', 'dim', 1) Actual Shape: (1, 1).
```

You can also name the variadic dimensions, to ensure that a contiguous sequence
of dimensions match between arguments. For example:

```python
@check_shapes('batch,variadic...', 'variadic...')
def h(a, b):
    pass

h(np.ones((7, 1, 2)), np.ones((1, 2)))  # succeeds
h(np.ones((6, 2)), np.ones((1, 1)))  # fails
h(np.ones((6, 2)), np.ones((1)))  # fails
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

If you have any questions or issues with the library, please raise an issue on
[GitHub](https://github.com/n2cholas/shapecheck/issues). Hope you enjoy using
the library!
