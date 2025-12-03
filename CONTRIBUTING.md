# Contributing to GAM-SSM-LUR

Thank you for your interest in contributing to GAM-SSM-LUR! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lur_space_state_model.git
   cd lur_space_state_model
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/GabrielOduori/lur_space_state_model.git
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify the setup**:
   ```bash
   pytest tests/
   ```

## Making Changes

### Branching Strategy

- Create a new branch for each feature or bugfix:
  ```bash
  git checkout -b feature/your-feature-name
  # or
  git checkout -b fix/your-bugfix-name
  ```

- Keep branches focused on a single change
- Regularly sync with upstream:
  ```bash
  git fetch upstream
  git rebase upstream/main
  ```

### Types of Contributions

We welcome:

- **Bug fixes**: Fix issues and improve stability
- **New features**: Add functionality that aligns with the project goals
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize code for speed or memory efficiency

## Code Style

We follow these conventions:

### Python Style

- **PEP 8** compliance (enforced by `ruff`)
- **Black** for code formatting (line length: 88)
- **Type hints** for all public functions
- **Docstrings** in NumPy style for all public modules, classes, and functions

### Example Function

```python
def compute_rmse(
    y_true: NDArray,
    y_pred: NDArray,
    weights: Optional[NDArray] = None,
) -> float:
    """Compute root mean square error.
    
    Parameters
    ----------
    y_true : NDArray
        Ground truth values, shape (n_samples,)
    y_pred : NDArray
        Predicted values, shape (n_samples,)
    weights : NDArray, optional
        Sample weights. If None, uniform weights are used.
        
    Returns
    -------
    float
        Root mean square error
        
    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> compute_rmse(y_true, y_pred)
    0.1
    """
    residuals = y_true - y_pred
    if weights is not None:
        return np.sqrt(np.average(residuals**2, weights=weights))
    return np.sqrt(np.mean(residuals**2))
```

### Running Linters

```bash
# Format code
black src/ tests/

# Check style
ruff check src/ tests/

# Type checking
mypy src/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gam_ssm_lur --cov-report=html

# Run specific test file
pytest tests/test_kalman.py

# Run specific test
pytest tests/test_kalman.py::test_filter_convergence
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_<module>.py`
- Name test functions as `test_<functionality>`
- Use fixtures from `conftest.py` for common setup
- Aim for high coverage of edge cases

### Test Example

```python
import pytest
import numpy as np
from gam_ssm_lur import KalmanFilter


class TestKalmanFilter:
    """Tests for KalmanFilter class."""
    
    @pytest.fixture
    def simple_filter(self):
        """Create a simple Kalman filter for testing."""
        kf = KalmanFilter(state_dim=2, obs_dim=2)
        kf.initialize(
            T=np.eye(2),
            Z=np.eye(2),
            Q=0.1 * np.eye(2),
            H=0.1 * np.eye(2),
        )
        return kf
        
    def test_filter_output_shape(self, simple_filter):
        """Test that filter output has correct shape."""
        observations = np.random.randn(100, 2)
        result = simple_filter.filter(observations)
        
        assert result.filtered_means.shape == (100, 2)
        assert result.filtered_covariances.shape == (100, 2, 2)
        
    def test_likelihood_increases(self, simple_filter):
        """Test that likelihood is computed correctly."""
        observations = np.random.randn(100, 2)
        result = simple_filter.filter(observations)
        
        assert np.isfinite(result.log_likelihood)
```

## Documentation

### Docstring Style

We use NumPy-style docstrings:

```python
"""Short description.

Longer description if needed, explaining the purpose
and behavior in more detail.

Parameters
----------
param1 : type
    Description of param1
param2 : type, optional
    Description of param2

Returns
-------
type
    Description of return value

Raises
------
ValueError
    When invalid input is provided

See Also
--------
related_function : Description of relationship

Notes
-----
Additional implementation notes.

References
----------
.. [1] Author, "Title", Journal, Year.

Examples
--------
>>> example_usage()
expected_output
"""
```

### Building Documentation

```bash
cd docs/
make html
# Open _build/html/index.html in browser
```

## Submitting Changes

### Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests and linters**:
   ```bash
   pytest
   black src/ tests/
   ruff check src/ tests/
   ```

3. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Include before/after examples if applicable

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] Tests pass locally
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive

### Commit Messages

Follow conventional commits:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(kalman): add block-diagonal mode for large networks`
- `fix(em): handle singular covariance matrices`
- `docs(readme): update installation instructions`

## Questions?

If you have questions, please:

1. Check existing issues and discussions
2. Open a new issue with the "question" label
3. Contact the maintainers

Thank you for contributing!
