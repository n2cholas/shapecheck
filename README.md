# ShapeCheck
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