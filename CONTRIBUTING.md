# Contributing to TPTBox

Thanks for helping improve this codebase. The following overview will help you as guide to contribute.

## Code Style

- **Line length**: 140 characters
- **Formatter**: Ruff (Black-compatible, double quotes)
- **Target Python**: 3.10+ syntax, but the package supports 3.9–3.14
- **Naming**: Ruff N-rules are largely ignored — mixed-case class/method names are acceptable in this codebase (medical domain conventions)
- **Complexity**: McCabe max=20; research code legitimately has deep branching
- `from __future__ import annotations` is used widely for forward references

## Get started!

Ready to contribute?

### 1) Create an <a href=https://github.com/Hendrik-code/TPTBox/issues>issue</a> on the GitHub repository

### 2) Fork the repository

### 3) Clone your fork

```bash
git clone https://github.com/<your-username>/TPTBox.git
cd TPTBox
```

### 4) Local install (alternative)

Make a venv (in whatever fashion you like), then:
```bash
pip install -e .
```

### 5) Ensure you install the pre-commit hook:
```bash
pre-commit install
```
This pre-commit hook will automatically fix some linting issues and block your commits so that every commit is clean in terms of formatting

### 6) Running checks locally

Before pushing, make sure all unit tests and ruff pass. Tests live in `unit_tests/` (not `TPTBox/tests/`). `TPTBox/tests/` contains test utilities and sample data (CT/MRI NIfTIs) used by the unit tests. Some test files are very large.

```bash
# All tests
pytest unit_tests/

# Single test file
pytest unit_tests/test_nii.py

# Single test function
pytest unit_tests/test_nii.py::test_function_name

# With coverage
coverage run --source=TPTBox -m pytest unit_tests/
```

Linting & Formatting:
```bash
# Lint (auto-fix where possible)
ruff check . --fix

# Format
ruff format .

# Both (mirrors pre-commit behavior)
pre-commit run --all-files
```




### 7) Submitting changes

1. Create a feature branch from `main`:

   ```bash
   git checkout -b my-feature
   ```

2. Make your changes and commit with a clear, descriptive message.
3. Push your branch and open a <a href=https://github.com/Hendrik-code/TPTBox/pulls>pull request</a> against `main`.
4. CI runs on all pull requests — ensure all checks are green before requesting review.
