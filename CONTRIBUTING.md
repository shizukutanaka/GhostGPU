# Contributing to GhostGPU

Thank you for your interest in contributing to GhostGPU! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Virtual environment (recommended)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/ghostgpu.git
cd ghostgpu

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

### Running Tests

```bash
# Run all tests
pytest tests/test_ghostgpu.py -v

# Run with coverage
pytest tests/test_ghostgpu.py --cov=ghost --cov-report=html

# Run specific test
pytest tests/test_ghostgpu.py::TestArrayCreation -v
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black ghost.py examples/ tests/

# Lint checks
flake8 ghost.py examples/ tests/

# Type checking
mypy ghost.py
```

## Development Workflow

### 1. Create an Issue

Before starting work, create an issue describing:
- What problem you're solving
- Your proposed solution
- Any relevant context

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/issue-description
```

### 3. Make Changes

- Keep commits focused and logical
- Write descriptive commit messages
- Add tests for new functionality
- Update documentation as needed

### 4. Commit Guidelines

```bash
# Good commit messages
git commit -m "Add array transpose support"
git commit -m "Fix memory leak in GPU transfer"
git commit -m "Improve documentation for broadcasting"

# Avoid vague messages
# ❌ git commit -m "fix stuff"
# ❌ git commit -m "WIP"
```

### 5. Testing Requirements

All new features must include tests:

```python
class TestNewFeature:
    """Test new feature functionality"""

    def test_basic_usage(self):
        """Test basic usage"""
        # Arrange
        arr = ghost.array([1, 2, 3])

        # Act
        result = ghost.new_operation(arr)

        # Assert
        assert result is not None
        arr.release()

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty arrays, zero values, etc.
        pass
```

### 6. Documentation

Update relevant documentation:
- Add docstrings to functions (Google style)
- Update README.md if adding public APIs
- Add examples if introducing new features

```python
def new_operation(array: GhostArray, parameter: int) -> GhostArray:
    """Brief description of operation.

    Longer description if needed. Explain behavior, special cases, etc.

    Args:
        array: Input array to process
        parameter: Configuration parameter

    Returns:
        Resulting GhostArray after operation

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> arr = ghost.array([1, 2, 3])
        >>> result = ghost.new_operation(arr, parameter=5)
        >>> arr.release()
    """
    pass
```

### 7. Performance Considerations

For performance-critical code:
- Profile before optimizing (`cProfile`)
- Benchmark changes
- Document performance implications
- Avoid premature optimization

### 8. Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub with:
# - Clear title
# - Description of changes
# - Reference to related issues
# - Test results
```

## PR Review Process

Pull requests will be reviewed for:
- ✅ Code quality (style, readability, type hints)
- ✅ Test coverage (>80% for new code)
- ✅ Documentation completeness
- ✅ Performance impact
- ✅ Security considerations

## Code Style

### Python Style Guide

We follow PEP 8 with some exceptions:
- Line length: 100 characters
- Use type hints for all functions
- Docstring style: Google

### Example

```python
from typing import Optional, Tuple
import numpy as np

def complex_operation(
    array1: 'GhostArray',
    array2: 'GhostArray',
    axis: Optional[int] = None,
) -> 'GhostArray':
    """Perform complex operation on two arrays.

    Args:
        array1: First input array
        array2: Second input array
        axis: Axis for operation (optional)

    Returns:
        Result array
    """
    pass
```

## Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Refactoring
- `perf`: Performance improvement
- `test`: Test additions
- `chore`: Maintenance

Examples:
```
feat: Add multi-GPU support

- Implement device detection
- Add distributed memory management
- Update tests for multi-device scenarios

Closes #123
```

## Reporting Bugs

When reporting bugs, include:
- Python version
- Operating system
- GhostGPU version
- Minimal reproduction code
- Expected vs actual behavior
- Error messages/tracebacks

```python
import ghost

# Minimal reproduction
arr = ghost.array([1, 2, 3])
result = ghost.problematic_operation(arr)  # Describe issue
arr.release()
```

## Requesting Features

Feature requests should include:
- Use case and motivation
- Proposed API/interface
- Example usage
- Alternative approaches considered

## Documentation

### Writing Docs

- Use clear, concise language
- Provide examples
- Include edge cases
- Link related content

### Building Docs

```bash
# If using Sphinx
cd docs
make html
```

## Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release notes
4. Tag commit: `git tag v1.0.0`
5. Push tag: `git push origin v1.0.0`

## Questions?

- Open a discussion on GitHub
- Check existing issues and PRs
- Review documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to GhostGPU!
