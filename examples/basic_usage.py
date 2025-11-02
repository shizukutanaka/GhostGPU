#!/usr/bin/env python3
"""
GhostGPU Basic Usage Examples
Demonstrates core functionality and best practices
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ghost
import numpy as np

def example_array_creation():
    """Example: Creating arrays"""
    print("\n" + "="*60)
    print("Example 1: Array Creation")
    print("="*60)

    # Create from list
    a = ghost.array([1, 2, 3, 4, 5])
    print(f"Array from list: {a.numpy()}")

    # Create zeros
    b = ghost.zeros((3, 3))
    print(f"Zeros array:\n{b.numpy()}")

    # Create ones
    c = ghost.ones((2, 4))
    print(f"Ones array:\n{c.numpy()}")

    # Create filled array
    d = ghost.full((2, 2), 7.5)
    print(f"Filled array:\n{d.numpy()}")

    # Cleanup
    a.release()
    b.release()
    c.release()
    d.release()

def example_elementwise_ops():
    """Example: Elementwise operations"""
    print("\n" + "="*60)
    print("Example 2: Elementwise Operations")
    print("="*60)

    a = ghost.array([1, 2, 3, 4])
    b = ghost.array([5, 6, 7, 8])

    # Addition
    c = ghost.add(a, b)
    print(f"Add: {a.numpy()} + {b.numpy()} = {c.numpy()}")

    # Multiplication
    d = ghost.multiply(a, b)
    print(f"Multiply: {a.numpy()} * {b.numpy()} = {d.numpy()}")

    # Power
    e = ghost.power(a, 2)
    print(f"Power: {a.numpy()} ** 2 = {e.numpy()}")

    # Square root
    f = ghost.sqrt(ghost.array([4.0, 9.0, 16.0, 25.0]))
    print(f"Sqrt: [4, 9, 16, 25] -> {f.numpy()}")

    # Cleanup
    for arr in [a, b, c, d, e, f]:
        arr.release()

def example_reductions():
    """Example: Reduction operations"""
    print("\n" + "="*60)
    print("Example 3: Reduction Operations")
    print("="*60)

    data = ghost.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Data:\n{data.numpy()}")

    # Sum all elements
    total = ghost.sum(data)
    print(f"Sum (all): {total}")

    # Sum along axis 0 (columns)
    sum_cols = ghost.sum(data, axis=0)
    print(f"Sum (axis=0): {sum_cols.numpy()}")

    # Sum along axis 1 (rows)
    sum_rows = ghost.sum(data, axis=1)
    print(f"Sum (axis=1): {sum_rows.numpy()}")

    # Mean
    avg = ghost.mean(data)
    print(f"Mean: {avg}")

    # Max and Min
    maximum = ghost.max(data)
    minimum = ghost.min(data)
    print(f"Max: {maximum}, Min: {minimum}")

    # Cleanup
    data.release()
    sum_cols.release()
    sum_rows.release()

def example_matrix_operations():
    """Example: Matrix operations"""
    print("\n" + "="*60)
    print("Example 4: Matrix Operations")
    print("="*60)

    # Matrix multiplication
    a = ghost.array([[1, 2], [3, 4]])
    b = ghost.array([[5, 6], [7, 8]])
    c = ghost.matmul(a, b)

    print(f"Matrix A:\n{a.numpy()}")
    print(f"Matrix B:\n{b.numpy()}")
    print(f"A @ B:\n{c.numpy()}")

    # Cleanup
    a.release()
    b.release()
    c.release()

def example_memory_management():
    """Example: Memory management with context manager"""
    print("\n" + "="*60)
    print("Example 5: Memory Management")
    print("="*60)

    # Using context manager (automatic cleanup)
    with ghost.zeros((1000, 1000)) as arr:
        print(f"Array shape: {arr.shape}")
        print(f"Array size: {arr.nbytes / (1024**2):.2f} MB")
        # Automatic cleanup when exiting context

    print("Array automatically released")

    # Manual memory management
    arr = ghost.ones((500, 500))
    print(f"Created array: {arr.shape}")
    arr.release()
    print("Manually released")

def example_broadcasting():
    """Example: Broadcasting operations"""
    print("\n" + "="*60)
    print("Example 6: Broadcasting")
    print("="*60)

    # Row vector + column vector
    row = ghost.array([[1, 2, 3]])
    col = ghost.array([[10], [20], [30]])

    result = ghost.add(row, col)
    print(f"Row: {row.numpy()}")
    print(f"Col:\n{col.numpy()}")
    print(f"Row + Col:\n{result.numpy()}")

    # Cleanup
    row.release()
    col.release()
    result.release()

def example_complex_computation():
    """Example: Complex computation pipeline"""
    print("\n" + "="*60)
    print("Example 7: Complex Computation Pipeline")
    print("="*60)

    # Compute: sqrt((a^2 + b^2) / 2) for arrays a and b
    a = ghost.array([3.0, 4.0, 5.0])
    b = ghost.array([4.0, 3.0, 12.0])

    # Step by step
    a_sq = ghost.power(a, 2)
    b_sq = ghost.power(b, 2)
    sum_sq = ghost.add(a_sq, b_sq)
    div = ghost.divide(sum_sq, 2.0)
    result = ghost.sqrt(div)

    print(f"a = {a.numpy()}")
    print(f"b = {b.numpy()}")
    print(f"sqrt((a^2 + b^2) / 2) = {result.numpy()}")

    # Cleanup
    for arr in [a, b, a_sq, b_sq, sum_sq, div, result]:
        arr.release()

def example_system_info():
    """Example: System information and health check"""
    print("\n" + "="*60)
    print("Example 8: System Information")
    print("="*60)

    info = ghost.get_info()
    print(f"Backend: {info['backend']}")
    print(f"Device Count: {info['device_count']}")
    print(f"Is Healthy: {info['is_healthy']}")

    if 'devices' in info:
        print("\nDevices:")
        for dev in info['devices']:
            print(f"  - {dev['name']}: {dev['memory_mb']} MB")

    # Health check
    health = ghost.health_check()
    print(f"\nHealth Status: {health['overall_status']}")

def example_benchmark():
    """Example: Running benchmarks"""
    print("\n" + "="*60)
    print("Example 9: Benchmark")
    print("="*60)

    # Run small benchmark
    results = ghost.benchmark(size=500)

    if results['success']:
        print(f"Backend: {results['backend']}")
        print(f"MatMul Performance: {results.get('matmul_gflops', 0):.2f} GFLOPS")
        print(f"Memory Bandwidth: {results.get('memory_bandwidth_gb_s', 0):.2f} GB/s")

def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("#  GhostGPU - Usage Examples")
    print("#"*60)

    try:
        example_array_creation()
        example_elementwise_ops()
        example_reductions()
        example_matrix_operations()
        example_memory_management()
        example_broadcasting()
        example_complex_computation()
        example_system_info()
        example_benchmark()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    finally:
        # Cleanup
        ghost.cleanup()

if __name__ == '__main__':
    main()
