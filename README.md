
# ShapeCheck

![Build & Tests](https://github.com/n2cholas/shapecheck/workflows/Build%20and%20Tests/badge.svg)

Framework-agnostic library for checking array shapes at runtime.

## Install Library

```bash
pip install --upgrade git+https://github.com/n2cholas/shapecheck.git
```

## Usage

```python
import numpy as np
from shapecheck import check_shape

@check_shape('-1,N', 'N', out='1,N')
def f(x, y):
    return (x + y).sum(0, keepdims=True)

f(np.ones((3, 5)), np.ones((5,)))  # succeeds
f(np.ones((7, 6)), np.ones((6,)))  # succeeds
f(np.ones((3, 2)), np.ones((5,)))  # fails
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

- Support non-array arguments/return values.
- Better error messages.
  - collect all issues then report instead of failing eagerly
  - provide argument name, expected shape, and given shape.
- Support "PyTrees" (nested dicts/tuples/lists of arrays)
- Support non-array inputs/outputs.
- Support variadic dimensions (via `...`).
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
