
# ShapeCheck

![Build & Tests](https://github.com/n2cholas/shapecheck/workflows/Build%20and%20Tests/badge.svg)

Framework-agnostic library for checking array shapes at runtime.

## Install Library

```bash
pip install --upgrade git+https://github.com/n2cholas/shapecheck.git
```

## Usage

```python
from shapecheck import check_shape

@check_shape((3,), (4,), out_=(2,))
def f(x, y):
    return x[:2]**2 + y[:2]**2

f(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))  # succeeds
f(np.array([1, 2]), np.array([1, 2, 3, 4]))  # fails
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

This library uses pre-commit hooks to fix formatting and other issues.
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
