
# ShapeCheck

![Build & Tests](https://github.com/n2cholas/shapecheck/workflows/Build%20and%20Tests/badge.svg)

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

@check_shapes('-1,N', 'N', None, '3,N', out='3,N')
def f(a, b, c, d):
    # a must be rank 2, where the first dim can be anything.
    # b must be rank 1, where the first dim must be N (a's second dim)
    # c will not be checked
    # d must be rank 2, where the first dim is 3, the second dim is N
    # since we specified `out=`, the output shape will be checked as well
    return (a + b).sum(0, keepdims=True) + d

f(np.ones((7, 5)), np.ones(5), 'anything', np.ones((3, 5)))  # succeeds
f(np.ones((2, 6)), np.ones(6), np.ones(1), np.ones((3, 6)))  # succeeds
f(np.ones((2, 6)), np.ones(5), np.ones(1), np.ones((3, 6)))  # fails
```

The last statement throws a `ShapeError` with an informative message.

```bash
shapecheck.shapecheck.ShapeError: in function f.
Named Dimensions: {'N': 6}.
Input:
    Match:    Argument: a Expected Shape: (-1, 'N') Actual Shape: (2, 6).
    MisMatch: Argument: b Expected Shape: ('N',) Actual Shape: (5,).
    Skipped:  Argument: c.
    Match:    Argument: d Expected Shape: (3, 'N') Actual Shape: (3, 6).
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

```bash
shapecheck.shapecheck.ShapeError: in function g.
Named Dimensions: {}.
Input:
    MisMatch: Argument: a Expected Shape: [1, '...', 1] Actual Shape: (2, 3, 4, 1).
    Match:    Argument: b Expected Shape: ['...', 1, 1] Actual Shape: (1, 1).
```

## Run Tests

```bash
git clone https://github.com/n2cholas/shapecheck.git
cd shapecheck
python setup.py develop
pip install -r requirements-dev.txt
pytest tests/
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
git add .
git commit -m "Make change"  # actually commits
```

If you don't use pre-commit, there is a GitHub action to automatically
format your code when you push to `main`.

## Planned Features

- Support "PyTrees" (nested dicts/tuples/lists of arrays)
- Provide context manager/switch to turn off shape checking.
- Support recursive checking (i.e. if parent and child function
  use named dimension 'N', ensure they're the same).

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
