
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
