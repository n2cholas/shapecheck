
# CONTRIBUTING

## Run Tests

```bash
git clone https://github.com/n2cholas/shapecheck.git
cd shapecheck
pip install --upgrade pip setuptools
python setup.py develop
pip install -r requirements-dev.txt
pytest tests/ -vvv --cov shapecheck --cov-report term-missing
```

## Pre-commit

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
