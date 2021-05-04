
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

This library has minimal overhead, and can be easily disabled globally after
prototyping if desired. With frameworks like JAX or TensorFlow, "runtime" is
actually "compile" or "trace" time, so you don't pay any cost during execution.
For frameworks like PyTorch, asynchronous execution will hide the cost of shape
checking. You only pay a small overhead with synchronous, eager frameworks like
numpy.

This library was inspired by many other tools, including
[nptyping](https://github.com/ramonhagenaars/nptyping),
[einops](https://github.com/arogozhnikov/einops),
[shapecheck](https://github.com/rosshemsley/shapecheck), [torch named
tensors](https://pytorch.org/docs/stable/named_tensor.html),
[tensorcheck](https://github.com/bodin-e/tensorcheck).

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

@check_shapes({'imgs': 'N,W,W,-1', 'labels': 'N,1'}, 'N', out_='')
def loss_fn(batch, weights):
    diff = (batch['imgs'].mean((1, 2, 3)) - batch['labels'].squeeze())
    return np.mean(weights * diff**2)

loss_fn({'imgs': np.ones((3, 2, 2, 1)), 'labels': np.ones((3, 1))},
        weights=np.ones(3))  # succeeds
loss_fn({'imgs': np.ones((5, 3, 3, 4)), 'labels': np.ones((5, 1))},
        weights=np.ones(5))  # succeeds
loss_fn({'imgs': np.ones((3, 5, 2, 1)), 'labels': np.ones((3, 1))},
        weights=np.ones(3))  # fails
```

Error message:

```
ShapeError: in function loss_fn.
Named Dimensions: {'N': 3, 'W': 5}.
Input:
    Argument: batch  Type: <class 'dict'>
        MisMatch: Key: imgs Expected Shape: ('N', 'W', 'W', -1) Actual Shape: (3, 5, 2, 1).
        Match:    Key: labels Expected Shape: ('N', 1) Actual Shape: (3, 1).
    Match:    Argument: weights Expected Shape: ('N',) Actual Shape: (3,).
```

In the above example, we compute the "loss" with a batch of data, which is a
dictionary with `'imgs'` and `'labels'`. We specify that `'imgs'` should be of
shape `N,W,W,-1`, i.e., `N` square images which can have any number of
channels (indicated by the `-1`).  We want the `'labels'` to have shape `N,1`,
and `weights` to be a vector of size `N`.

Finally, we specify the output shape should be a scalar via `out_=''`. All
non-input shape arguments to `check_shape` have an underscore after them so
they don't conflict with the decorated function's arguments (for now, just
`out_` and `match_callees_`).

Inputs to `check_shape` can be arbitrarily nested dicts/lists/tuples, as long
as the structure of the shape specification matches the structure of the inputs
to the decorated function.

The shapes can be specified as positional arguments (like in the example), or
as key word arguments. For example, the following two are equivalent to the
example above:

```python
@check_shapes(batch={'imgs': 'N,W,W,-1', 'labels': 'N,1'},
              weights='N', out_='')
def loss_fn(batch, weights):
    ...

@check_shapes({'imgs': 'N,W,W,-1', 'labels': 'N,1'},
              weights='N', out_='')
def loss_fn(batch, weights):
    ...
```

### Skipping Arguments

If you want to not check an argument (e.g. if it's not an array), you can
exclude that argument, or specify that the shape is `None`. For example, the
following two are equivalent:

```python
@check_shapes(None, 'N,2')
def fn(arg1_that_isnt_checked, arg2):
    ...

@check_shapes(arg2='N,2')
def fn(arg1_that_isnt_checked, arg2):
    ...
```

### Consistent Shapes between Functionc Calls

If you have a function with shape-checking that calls other functions with
shape-checking, you can optionally enforce that dimensions with the same letter
name in the caller correspond to the same sized dimension in the callees via
`match_callees_=True`.  That is, you can check that named dimensions are
consistent between functions on the call stack.

```python
@check_shapes('M')
def fn2(x):
    return x

@check_shapes('M', 'N', match_callees_=True)
def fn1(x, y):
    return fn2(y)

fn1(np.ones(5), np.ones(6))  # fails
```

Here, we (mistakenly) used `y` instead of `x` when calling `fn2`.  When
`match_callees_=False`, this function would work just fine.  When
`match_callees_=True`, we get the following error:

```
ShapeError: in function fn2.
Named Dimensions: {'M': 5, 'N': 6}.
Input:
    MisMatch: Argument: a Expected Shape: ('M',) Actual Shape: (6,).
```

### Variadic Dimensions

This library also supports variadic dimensions. You can use `'...'` to indicate
0 or more dimensions (the spaces in the example strings are optional):

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
ShapeError: in function g.
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
h(np.ones((6, 2, 1)), np.ones((1, 1)))  # fails
h(np.ones((6, 2)), np.ones((1)))  # fails
```

The error message for the first fail would be:

```
ShapeError: in function h.
Named Dimensions: {'batch': 6, 'variadic...': (2, 1)}.
Input:
    Match:    Argument: a Expected Shape: ('batch', 'variadic...') Actual Shape: (6, 2, 1).
    MisMatch: Argument: b Expected Shape: ('variadic...',) Actual Shape: (1, 1).
```

### Toggling Checking Globally

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
