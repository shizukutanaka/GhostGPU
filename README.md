# GhostGPU

A lightweight numerical computing runtime for CPU and GPU backends with a focus on simplicity, reliability, and extensibility.

## Overview

GhostGPU provides a NumPy-like API for numerical operations with automatic memory management and support for both CPU and GPU execution. It's designed to be a dependable foundation for scientific computing with optional hardware acceleration.

## Key Features

- **NumPy-Compatible API**: Familiar interface for array operations
- **CPU-Backed Core**: Dependable performance with automatic fallback
- **Memory Management**: Built-in memory accounting and safe cleanup
- **Array Operations**: Comprehensive support for elementwise and reduction operations
- **Matrix Operations**: Full support for matrix multiplication and decomposition
- **Optional GPU Support**: Seamless GPU acceleration when available
- **Health Checks**: System diagnostics and performance benchmarking
- **Serialization**: Safe data serialization with encryption support

## Installation

### Basic Installation

```bash
pip install numpy>=1.20.0
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -r requirements.txt[dev]
```

## Quick Start

```python
import ghost
import numpy as np

# Create arrays
a = ghost.array([1, 2, 3, 4])
b = ghost.zeros((3, 3))
c = ghost.ones((2, 4))

# Elementwise operations
result = ghost.add(a, ghost.array([5, 6, 7, 8]))

# Reductions
total = ghost.sum(result)
average = ghost.mean(result)

# Matrix operations
m1 = ghost.array([[1, 2], [3, 4]])
m2 = ghost.array([[5, 6], [7, 8]])
m_result = ghost.matmul(m1, m2)

# Cleanup
a.release()
result.release()
```

## Core API

### Array Creation

```python
ghost.array(data)           # Create from list/array
ghost.zeros(shape)          # Create zero array
ghost.ones(shape)           # Create ones array
ghost.empty(shape)          # Create uninitialized array
ghost.full(shape, value)    # Create filled array
ghost.arange(start, stop)   # Create range array
```

### Elementwise Operations

```python
ghost.add(a, b)
ghost.subtract(a, b)
ghost.multiply(a, b)
ghost.divide(a, b)
ghost.power(a, exponent)
ghost.sqrt(a)
ghost.exp(a)
ghost.log(a)
ghost.abs(a)
```

### Reduction Operations

```python
ghost.sum(array, axis=None)
ghost.mean(array, axis=None)
ghost.max(array, axis=None)
ghost.min(array, axis=None)
ghost.std(array, axis=None)
```

### Matrix Operations

```python
ghost.matmul(a, b)          # Matrix multiplication
ghost.transpose(a)          # Transpose
ghost.dot(a, b)             # Dot product
```

### Memory Management

```python
array.release()             # Manual cleanup
with ghost.array(...) as a: # Automatic cleanup
    ...
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_ghostgpu.py -v
```

Run specific test class:

```bash
pytest tests/test_ghostgpu.py::TestArrayCreation -v
```

## Examples

See `examples/basic_usage.py` for comprehensive usage examples including:

- Array creation and initialization
- Elementwise operations
- Reduction operations
- Matrix operations
- Memory management with context managers
- Broadcasting
- Complex computation pipelines
- System information and diagnostics
- Performance benchmarking

Run examples:

```bash
python examples/basic_usage.py
```

## Architecture

### Backend Design

GhostGPU uses a modular backend system:

- **CPUBackend**: NumPy-based fallback for universal compatibility
- **GPUBackend**: CuPy support for NVIDIA CUDA devices
- **Backend Interface**: Pluggable architecture for custom implementations

### Memory Model

- **Pinned Memory**: Efficient GPU transfers with optional page-locking
- **Memory Accounting**: Track memory usage per array
- **Lazy Cleanup**: Safe deallocation with reference counting

### Type System

- Full type hints for IDE support and type checking
- Runtime type validation for operation safety
- Generic support for multiple numeric types

## Performance

### Benchmarking

```python
results = ghost.benchmark(size=1000)
print(f"MatMul: {results['matmul_gflops']:.2f} GFLOPS")
print(f"Bandwidth: {results['memory_bandwidth_gb_s']:.2f} GB/s")
```

### Profiling

```python
# Profile specific operations
prof = ghost.profile_matmul(size=500, backend='cpu')
prof.print_stats()
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Citation

If you use GhostGPU in your research, please cite:

```bibtex
@software{ghostgpu2025,
  title={GhostGPU: A Lightweight Numerical Computing Runtime},
  author={GhostGPU Contributors},
  year={2025},
  url={https://github.com/your-org/ghostgpu}
}
```

## Support

- Documentation: [docs/](docs/)
- Issue Tracker: GitHub Issues
- Discussions: GitHub Discussions

## Roadmap

- [ ] Multi-GPU support
- [ ] Custom kernel support
- [ ] Advanced memory optimization
- [ ] Distributed computing support
- [ ] Additional hardware backends (AMD, Intel)
