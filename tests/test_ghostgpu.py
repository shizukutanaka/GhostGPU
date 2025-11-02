#!/usr/bin/env python3
"""
Comprehensive test suite for GhostGPU
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ghost

class TestArrayCreation:
    """Test array creation functions"""

    def test_array(self):
        arr = ghost.array([1, 2, 3, 4])
        np_arr = np.asarray(arr)
        assert np_arr.shape == (4,)
        assert np.allclose(np_arr, [1, 2, 3, 4])
        arr.release()

    def test_zeros(self):
        arr = ghost.zeros((3, 3))
        np_arr = np.asarray(arr)
        assert np_arr.shape == (3, 3)
        assert np.all(np_arr == 0)
        arr.release()

    def test_ones(self):
        arr = ghost.ones((2, 4))
        np_arr = np.asarray(arr)
        assert np_arr.shape == (2, 4)
        assert np.all(np_arr == 1)
        arr.release()

    def test_empty(self):
        arr = ghost.empty((5, 5))
        np_arr = np.asarray(arr)
        assert np_arr.shape == (5, 5)
        arr.release()

    def test_full(self):
        arr = ghost.full((3, 3), 7.5)
        np_arr = np.asarray(arr)
        assert np_arr.shape == (3, 3)
        assert np.all(np_arr == 7.5)
        arr.release()

class TestElementwiseOperations:
    """Test elementwise operations"""

    def test_add(self):
        a = ghost.array([1, 2, 3])
        b = ghost.array([4, 5, 6])
        c = ghost.add(a, b)
        assert np.allclose(np.asarray(c), [5, 7, 9])
        a.release()
        b.release()
        c.release()

    def test_subtract(self):
        a = ghost.array([10, 20, 30])
        b = ghost.array([1, 2, 3])
        c = ghost.subtract(a, b)
        assert np.allclose(np.asarray(c), [9, 18, 27])
        a.release()
        b.release()
        c.release()

    def test_multiply(self):
        a = ghost.array([2, 3, 4])
        b = ghost.array([5, 6, 7])
        c = ghost.multiply(a, b)
        assert np.allclose(np.asarray(c), [10, 18, 28])
        a.release()
        b.release()
        c.release()

    def test_divide(self):
        a = ghost.array([10.0, 20.0, 30.0])
        b = ghost.array([2.0, 4.0, 5.0])
        c = ghost.divide(a, b)
        assert np.allclose(np.asarray(c), [5.0, 5.0, 6.0])
        a.release()
        b.release()
        c.release()

    def test_power(self):
        a = ghost.array([2, 3, 4])
        c = ghost.power(a, 2)
        assert np.allclose(np.asarray(c), [4, 9, 16])
        a.release()
        c.release()

    def test_sqrt(self):
        a = ghost.array([4.0, 9.0, 16.0])
        c = ghost.sqrt(a)
        assert np.allclose(np.asarray(c), [2.0, 3.0, 4.0])
        a.release()
        c.release()

    def test_exp(self):
        a = ghost.array([0.0, 1.0, 2.0])
        c = ghost.exp(a)
        expected = np.exp([0.0, 1.0, 2.0])
        assert np.allclose(np.asarray(c), expected)
        a.release()
        c.release()

    def test_log(self):
        a = ghost.array([1.0, np.e, np.e**2])
        c = ghost.log(a)
        assert np.allclose(np.asarray(c), [0.0, 1.0, 2.0])
        a.release()
        c.release()

class TestReductions:
    """Test reduction operations"""

    def test_sum(self):
        a = ghost.array([[1, 2, 3], [4, 5, 6]])
        total = ghost.sum(a)
        assert total == 21
        a.release()

    def test_sum_axis0(self):
        a = ghost.array([[1, 2, 3], [4, 5, 6]])
        result = ghost.sum(a, axis=0)
        assert np.allclose(np.asarray(result), [5, 7, 9])
        a.release()
        result.release()

    def test_sum_axis1(self):
        a = ghost.array([[1, 2, 3], [4, 5, 6]])
        result = ghost.sum(a, axis=1)
        assert np.allclose(np.asarray(result), [6, 15])
        a.release()
        result.release()

    def test_mean(self):
        a = ghost.array([[2, 4, 6], [8, 10, 12]])
        avg = ghost.mean(a)
        assert np.isclose(avg, 7.0)
        a.release()

    def test_max(self):
        a = ghost.array([[1, 5, 3], [9, 2, 7]])
        maximum = ghost.max(a)
        assert maximum == 9
        a.release()

    def test_min(self):
        a = ghost.array([[5, 3, 8], [2, 9, 1]])
        minimum = ghost.min(a)
        assert minimum == 1
        a.release()

class TestStatistics:
    """Test statistical functions"""

    def test_std(self):
        a = ghost.array([1, 2, 3, 4, 5])
        std_val = ghost.std(a)
        expected = np.std([1, 2, 3, 4, 5])
        assert np.isclose(std_val, expected)
        a.release()

    def test_var(self):
        a = ghost.array([1, 2, 3, 4, 5])
        var_val = ghost.var(a)
        expected = np.var([1, 2, 3, 4, 5])
        assert np.isclose(var_val, expected)
        a.release()

    def test_std_axis0(self):
        a = ghost.array([[1, 2, 3], [4, 5, 6]])
        std_val = ghost.std(a, axis=0)
        expected = np.std([[1, 2, 3], [4, 5, 6]], axis=0)
        assert np.allclose(np.asarray(std_val), expected)
        a.release()
        std_val.release()

    def test_var_axis1(self):
        a = ghost.array([[1, 2, 3], [4, 5, 6]])
        var_val = ghost.var(a, axis=1)
        expected = np.var([[1, 2, 3], [4, 5, 6]], axis=1)
        assert np.allclose(np.asarray(var_val), expected)
        a.release()
        var_val.release()

class TestUtilities:
    """Test utility functions"""

    def test_linspace_basic(self):
        result = ghost.linspace(0, 10, 5)
        expected = np.linspace(0, 10, 5)
        assert np.allclose(np.asarray(result), expected)
        result.release()

    def test_linspace_no_endpoint(self):
        result = ghost.linspace(0, 10, 5, endpoint=False)
        expected = np.linspace(0, 10, 5, endpoint=False)
        assert np.allclose(np.asarray(result), expected)
        result.release()

    def test_arange_basic(self):
        result = ghost.arange(5)
        expected = np.arange(5)
        assert np.allclose(np.asarray(result), expected)
        result.release()

    def test_arange_start_stop(self):
        result = ghost.arange(1, 10, 2)
        expected = np.arange(1, 10, 2)
        assert np.allclose(np.asarray(result), expected)
        result.release()

    def test_arange_negative_step(self):
        result = ghost.arange(10, 0, -1)
        expected = np.arange(10, 0, -1)
        assert np.allclose(np.asarray(result), expected)
        result.release()

class TestMemoryManagement:
    """Test memory management and utilities"""

    def test_memory_stats(self):
        stats = ghost.get_memory_stats()
        assert isinstance(stats, dict)
        assert 'current_allocated_mb' in stats
        assert 'peak_allocated_mb' in stats
        assert 'num_live_objects' in stats

    def test_memory_limit(self):
        # Test setting and getting memory limit
        original_limit = ghost.get_memory_limit()
        try:
            ghost.set_memory_limit(1000.0)
            assert ghost.get_memory_limit() == 1000.0

            ghost.set_memory_limit(0.0)
            assert ghost.get_memory_limit() == 0.0
        finally:
            # Restore original limit
            ghost.set_memory_limit(original_limit)

    def test_memory_limit_negative(self):
        with pytest.raises(ValueError):
            ghost.set_memory_limit(-100.0)

    def test_force_gc(self):
        # Test that force_gc doesn't raise exceptions
        ghost.force_gc()

class TestArrayOperations:
    """Test array manipulation operations"""

    def test_tile(self):
        a = ghost.array([1, 2, 3])
        b = ghost.tile(a, 2)
        expected = np.tile([1, 2, 3], 2)
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_tile_2d(self):
        a = ghost.array([[1, 2], [3, 4]])
        b = ghost.tile(a, (2, 3))
        expected = np.tile([[1, 2], [3, 4]], (2, 3))
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_repeat(self):
        a = ghost.array([1, 2, 3])
        b = ghost.repeat(a, 2)
        expected = np.repeat([1, 2, 3], 2)
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_repeat_axis(self):
        a = ghost.array([[1, 2], [3, 4]])
        b = ghost.repeat(a, 2, axis=0)
        expected = np.repeat([[1, 2], [3, 4]], 2, axis=0)
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_clip(self):
        a = ghost.array([1, 5, 10, 15, 20])
        b = ghost.clip(a, 5, 15)
        expected = np.clip([1, 5, 10, 15, 20], 5, 15)
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_clip_no_bounds(self):
        a = ghost.array([1, 5, 10, 15, 20])
        b = ghost.clip(a, None, 15)
        expected = np.clip([1, 5, 10, 15, 20], None, 15)
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

class TestPerformance:
    """Test performance profiling and optimization"""

    def test_profile_operation(self):
        a = ghost.array([1, 2, 3])
        b = ghost.array([4, 5, 6])

        profile = ghost.profile_operation('test_add', lambda: ghost.add(a, b))

        assert profile['operation'] == 'test_add'
        assert 'duration_ms' in profile
        assert profile['success'] is True
        assert 'result' in profile
        assert 'timestamp' in profile

        # Cleanup
        a.release()
        b.release()
        if hasattr(profile['result'], 'release'):
            profile['result'].release()

    def test_benchmark_operations(self):
        results = ghost.benchmark_operations(sizes=[50], operations=['add'])

        assert 'sizes' in results
        assert 'operations' in results
        assert 'results' in results
        assert 50 in results['results']
        assert 'add' in results['results'][50]

    def test_optimize_kernel_config_matmul(self):
        config = ghost.optimize_kernel_config((100, 100), 'matmul')

        assert 'block_size' in config
        assert 'grid_size' in config
        assert 'shared_memory' in config
        assert 'notes' in config

    def test_optimize_kernel_config_elementwise(self):
        config = ghost.optimize_kernel_config((1000,), 'elementwise')

        assert 'block_size' in config
        assert 'grid_size' in config
        assert 'shared_memory' in config
        assert 'notes' in config

class TestSortingSearch:
    """Test sorting and search functions"""

    def test_sort(self):
        a = ghost.array([3, 1, 4, 1, 5])
        b = ghost.sort(a)
        expected = np.sort([3, 1, 4, 1, 5])
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_sort_2d(self):
        a = ghost.array([[3, 1], [4, 1]])
        b = ghost.sort(a, axis=0)
        expected = np.sort([[3, 1], [4, 1]], axis=0)
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_argsort(self):
        a = ghost.array([3, 1, 4, 1, 5])
        b = ghost.argsort(a)
        expected = np.argsort([3, 1, 4, 1, 5])
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_argmax(self):
        a = ghost.array([3, 1, 4, 1, 5])
        idx = ghost.argmax(a)
        expected = np.argmax([3, 1, 4, 1, 5])
        assert idx == expected
        a.release()

    def test_argmax_axis(self):
        a = ghost.array([[1, 2, 3], [6, 5, 4]])
        b = ghost.argmax(a, axis=0)
        expected = np.argmax([[1, 2, 3], [6, 5, 4]], axis=0)
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_argmin(self):
        a = ghost.array([3, 1, 4, 1, 5])
        idx = ghost.argmin(a)
        expected = np.argmin([3, 1, 4, 1, 5])
        assert idx == expected
        a.release()

    def test_all(self):
        a = ghost.array([True, True, True])
        result = ghost.all(a)
        assert result == True

        b = ghost.array([True, False, True])
        result = ghost.all(b)
        assert result == False

        a.release()
        b.release()

    def test_all_axis(self):
        a = ghost.array([[True, False], [True, True]])
        b = ghost.all(a, axis=0)
        expected = np.all([[True, False], [True, True]], axis=0)
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_any(self):
        a = ghost.array([False, False, False])
        result = ghost.any(a)
        assert result == False

        b = ghost.array([False, True, False])
        result = ghost.any(b)
        assert result == True

        a.release()
        b.release()

class TestComparisonSet:
    """Test comparison and set functions"""

    def test_maximum(self):
        a = ghost.array([1, 5, 3])
        b = ghost.array([2, 4, 6])
        c = ghost.maximum(a, b)
        expected = np.maximum([1, 5, 3], [2, 4, 6])
        assert np.allclose(np.asarray(c), expected)
        a.release()
        b.release()
        c.release()

    def test_minimum(self):
        a = ghost.array([1, 5, 3])
        b = ghost.array([2, 4, 6])
        c = ghost.minimum(a, b)
        expected = np.minimum([1, 5, 3], [2, 4, 6])
        assert np.allclose(np.asarray(c), expected)
        a.release()
        b.release()
        c.release()

    def test_unique(self):
        a = ghost.array([1, 2, 2, 3, 1, 4])
        b = ghost.unique(a)
        expected = np.unique([1, 2, 2, 3, 1, 4])
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

    def test_unique_with_counts(self):
        a = ghost.array([1, 2, 2, 3, 1, 4])
        unique_vals, counts = ghost.unique(a, return_counts=True)
        expected_vals, expected_counts = np.unique([1, 2, 2, 3, 1, 4], return_counts=True)
        assert np.allclose(np.asarray(unique_vals), expected_vals)
        assert np.allclose(np.asarray(counts), expected_counts)
        a.release()
        unique_vals.release()
        counts.release()

    def test_where_condition_only(self):
        condition = ghost.array([True, False, True, False])
        indices = ghost.where(condition)
        expected = np.where([True, False, True, False])
        assert len(indices) == len(expected)
        for i, idx in enumerate(indices):
            assert np.allclose(np.asarray(idx), expected[i])
        condition.release()
        for idx in indices:
            idx.release()

    def test_where_with_xy(self):
        condition = ghost.array([True, False, True, False])
        x = ghost.array([1, 2, 3, 4])
        y = ghost.array([10, 20, 30, 40])
        result = ghost.where(condition, x, y)
        expected = np.where([True, False, True, False], [1, 2, 3, 4], [10, 20, 30, 40])
        assert np.allclose(np.asarray(result), expected)
        condition.release()
        x.release()
        y.release()
        result.release()

class TestFileIO:
    """Test file I/O functions"""

    def test_save_load(self, tmp_path):
        test_file = tmp_path / "test_array.npy"

        # Create test array
        original = ghost.array([1, 2, 3, 4, 5])
        ghost.save(str(test_file), original)

        # Load array
        loaded = ghost.load(str(test_file))

        # Check equality
        assert np.allclose(np.asarray(original), np.asarray(loaded))

        # Cleanup
        original.release()
        loaded.release()

    def test_save_load_float(self, tmp_path):
        test_file = tmp_path / "test_float.npy"

        # Create test array with floats
        original = ghost.array([1.1, 2.2, 3.3, 4.4, 5.5])
        ghost.save(str(test_file), original)

        # Load array
        loaded = ghost.load(str(test_file))

        # Check equality
        assert np.allclose(np.asarray(original), np.asarray(loaded))

        # Cleanup
        original.release()
        loaded.release()

class TestRandom:
    """Test random number generation functions"""

    def test_random_rand(self):
        result = ghost.random.rand(5)
        assert result.shape == (5,)
        assert np.all((result >= 0) & (result < 1))
        result.release()

    def test_random_rand_2d(self):
        result = ghost.random.rand(3, 4)
        assert result.shape == (3, 4)
        assert np.all((result >= 0) & (result < 1))
        result.release()

    def test_random_randn(self):
        result = ghost.random.randn(10)
        assert result.shape == (10,)
        # Check that values are reasonable for normal distribution
        assert abs(np.mean(np.asarray(result))) < 0.5  # Should be close to 0
        assert abs(np.std(np.asarray(result)) - 1.0) < 0.5  # Should be close to 1
        result.release()

    def test_random_randint(self):
        result = ghost.random.randint(0, 10, size=5)
        assert result.shape == (5,)
        assert np.all((result >= 0) & (result < 10))
        result.release()

    def test_random_uniform(self):
        result = ghost.random.uniform(5, 10, size=10)
        assert result.shape == (10,)
        assert np.all((result >= 5) & (result < 10))
        result.release()

    def test_random_normal(self):
        result = ghost.random.normal(5, 2, size=20)
        assert result.shape == (20,)
        # Check that mean is reasonable
        mean = np.mean(np.asarray(result))
        assert abs(mean - 5) < 1.0
        result.release()

    def test_random_seed(self):
        # Test reproducibility with seed
        ghost.random.seed(42)
        result1 = ghost.random.rand(3)

        ghost.random.seed(42)
        result2 = ghost.random.rand(3)

        assert np.allclose(np.asarray(result1), np.asarray(result2))

        result1.release()
        result2.release()

class TestInterpolationCalculus:
    """Test interpolation and calculus functions"""

    def test_interp(self):
        xp = ghost.array([1, 2, 3, 4, 5])
        fp = ghost.array([10, 20, 30, 40, 50])
        x = ghost.array([1.5, 2.5, 3.5])
        result = ghost.interp(x, xp, fp)
        expected = np.interp([1.5, 2.5, 3.5], [1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert np.allclose(np.asarray(result), expected)
        xp.release()
        fp.release()
        x.release()
        result.release()

    def test_gradient(self):
        a = ghost.array([1, 2, 4, 7, 11, 16])
        result = ghost.gradient(a)
        expected = np.gradient([1, 2, 4, 7, 11, 16])
        assert np.allclose(np.asarray(result), expected)
        a.release()
        result.release()

    def test_gradient_2d(self):
        a = ghost.array([[1, 2, 6], [3, 5, 9]])
        result = ghost.gradient(a)
        expected = np.gradient([[1, 2, 6], [3, 5, 9]])
        assert len(result) == len(expected)
        for i in range(len(result)):
            assert np.allclose(np.asarray(result[i]), expected[i])
        a.release()
        for r in result:
            r.release()

    def test_trapz(self):
        y = ghost.array([1, 2, 3, 4, 5])
        result = ghost.trapz(y)
        expected = np.trapz([1, 2, 3, 4, 5])
        assert abs(result - expected) < 1e-10
        y.release()

    def test_trapz_with_x(self):
        y = ghost.array([1, 2, 3, 4, 5])
        x = ghost.array([0, 1, 2, 3, 4])
        result = ghost.trapz(y, x)
        expected = np.trapz([1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
        assert abs(result - expected) < 1e-10
        y.release()
        x.release()

    def test_cumsum(self):
        a = ghost.array([1, 2, 3, 4])
        result = ghost.cumsum(a)
        expected = np.cumsum([1, 2, 3, 4])
        assert np.allclose(np.asarray(result), expected)
        a.release()
        result.release()

    def test_cumprod(self):
        a = ghost.array([1, 2, 3, 4])
        result = ghost.cumprod(a)
        expected = np.cumprod([1, 2, 3, 4])
        assert np.allclose(np.asarray(result), expected)
        a.release()
        result.release()

    def test_diff(self):
        a = ghost.array([1, 2, 4, 7, 0])
        result = ghost.diff(a)
        expected = np.diff([1, 2, 4, 7, 0])
        assert np.allclose(np.asarray(result), expected)
        a.release()
        result.release()

    def test_diff_n2(self):
        a = ghost.array([1, 2, 4, 7, 11])
        result = ghost.diff(a, n=2)
        expected = np.diff([1, 2, 4, 7, 11], n=2)
        assert np.allclose(np.asarray(result), expected)
        a.release()
        result.release()

class TestSignalProcessing:
    """Test signal processing and utility functions"""

    def test_convolve(self):
        a = ghost.array([1, 2, 3])
        v = ghost.array([0, 1, 0.5])
        result = ghost.convolve(a, v)
        expected = np.convolve([1, 2, 3], [0, 1, 0.5])
        assert np.allclose(np.asarray(result), expected)
        a.release()
        v.release()
        result.release()

    def test_correlate(self):
        a = ghost.array([1, 2, 3])
        v = ghost.array([0, 1, 0.5])
        result = ghost.correlate(a, v)
        expected = np.correlate([1, 2, 3], [0, 1, 0.5])
        assert np.allclose(np.asarray(result), expected)
        a.release()
        v.release()
        result.release()

    def test_polyval(self):
        p = ghost.array([1, 2, 3])  # 1 + 2x + 3x^2
        x = ghost.array([0, 1, 2])
        result = ghost.polyval(p, x)
        expected = np.polyval([1, 2, 3], [0, 1, 2])
        assert np.allclose(np.asarray(result), expected)
        p.release()
        x.release()
        result.release()

    def test_roll(self):
        a = ghost.array([1, 2, 3, 4, 5])
        result = ghost.roll(a, 2)
        expected = np.roll([1, 2, 3, 4, 5], 2)
        assert np.allclose(np.asarray(result), expected)
        a.release()
        result.release()

    def test_flip(self):
        a = ghost.array([1, 2, 3, 4, 5])
        result = ghost.flip(a)
        expected = np.flip([1, 2, 3, 4, 5])
        assert np.allclose(np.asarray(result), expected)
        a.release()
        result.release()

class TestMatrixOperations:
    """Test matrix operations"""

    def test_matmul_basic(self):
        a = ghost.array([[1, 2], [3, 4]])
        b = ghost.array([[5, 6], [7, 8]])
        c = ghost.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(np.asarray(c), expected)
        a.release()
        b.release()
        c.release()

    def test_matmul_rectangular(self):
        a = ghost.array([[1, 2, 3], [4, 5, 6]])
        b = ghost.array([[7, 8], [9, 10], [11, 12]])
        c = ghost.matmul(a, b)
        assert c.shape == (2, 2)
        a.release()
        b.release()
        c.release()

    def test_matmul_invalid_shape(self):
        a = ghost.array([[1, 2], [3, 4]])
        b = ghost.array([[5, 6, 7]])
        with pytest.raises(ValueError):
            ghost.matmul(a, b)
        a.release()
        b.release()

class TestMemoryManagement:
    """Test memory management"""

    def test_managed_array_context(self):
        with ghost.zeros((10, 10)) as arr:
            assert arr.shape == (10, 10)
        assert arr.is_released()

    def test_array_copy(self):
        a = ghost.array([1, 2, 3, 4])
        b = a.copy()
        assert np.allclose(a.numpy(), b.numpy())
        a.release()
        b.release()

    def test_asarray_no_copy(self):
        a = ghost.array([1, 2, 3])
        b = ghost.asarray(a, copy=False)
        # Should be same object when no copy
        assert np.shares_memory(a.numpy(), b.numpy())
        a.release()

    def test_asarray_with_copy(self):
        a = ghost.array([1, 2, 3])
        b = ghost.asarray(a, copy=True)
        assert not np.shares_memory(a.numpy(), b.numpy())
        a.release()
        b.release()

class TestSystemFunctions:
    """Test system functions"""

    def test_get_info(self):
        info = ghost.get_info()
        assert 'ghostgpu_version' in info
        assert 'backend' in info
        assert 'device_count' in info
        assert 'is_healthy' in info

    def test_health_check(self):
        health = ghost.health_check()
        assert 'overall_status' in health
        assert 'checks' in health
        assert health['overall_status'] in ['healthy', 'degraded', 'unhealthy']

    def test_benchmark(self):
        results = ghost.benchmark(size=100)
        assert results['success'] is True
        assert 'matmul_gflops' in results
        assert 'backend' in results

class TestDtypeSupport:
    """Test different data types"""

    def test_float32(self):
        a = ghost.array([1, 2, 3], dtype=np.float32)
        assert a.dtype == np.float32
        a.release()

    def test_float64(self):
        a = ghost.array([1, 2, 3], dtype=np.float64)
        assert a.dtype == np.float64
        a.release()

    def test_int32(self):
        a = ghost.array([1, 2, 3], dtype=np.int32)
        assert a.dtype == np.int32
        a.release()

class TestArrayManipulation:
    """Test array manipulation operations"""

    def test_reshape(self):
        a = ghost.array([1, 2, 3, 4, 5, 6])
        b = ghost.reshape(a, (2, 3))
        assert b.shape == (2, 3)
        a.release()
        b.release()

    def test_transpose(self):
        a = ghost.array([[1, 2, 3], [4, 5, 6]])
        b = ghost.transpose(a)
        assert b.shape == (3, 2)
        assert np.allclose(np.asarray(b), [[1, 4], [2, 5], [3, 6]])
        a.release()
        b.release()

    def test_concatenate(self):
        a = ghost.array([1, 2, 3])
        b = ghost.array([4, 5, 6])
        c = ghost.concatenate([a, b], axis=0)
        assert np.allclose(np.asarray(c), [1, 2, 3, 4, 5, 6])
        a.release()
        b.release()
        c.release()

    def test_stack(self):
        a = ghost.array([1, 2, 3])
        b = ghost.array([4, 5, 6])
        c = ghost.stack([a, b], axis=0)
        assert c.shape == (2, 3)
        a.release()
        b.release()
        c.release()

    def test_clip(self):
        a = ghost.array([1, 5, 10, 15, 20])
        b = ghost.clip(a, 5, 15)
        assert np.allclose(np.asarray(b), [5, 5, 10, 15, 15])
        a.release()
        b.release()

class TestTrigonometric:
    """Test trigonometric functions"""

    def test_sin(self):
        a = ghost.array([0.0, np.pi/2, np.pi])
        b = ghost.sin(a)
        assert np.allclose(np.asarray(b), [0.0, 1.0, 0.0], atol=1e-6)
        a.release()
        b.release()

    def test_cos(self):
        a = ghost.array([0.0, np.pi/2, np.pi])
        b = ghost.cos(a)
        assert np.allclose(np.asarray(b), [1.0, 0.0, -1.0], atol=1e-6)
        a.release()
        b.release()

    def test_tanh(self):
        a = ghost.array([0.0, 1.0, -1.0])
        b = ghost.tanh(a)
        expected = np.tanh([0.0, 1.0, -1.0])
        assert np.allclose(np.asarray(b), expected)
        a.release()
        b.release()

class TestSignalProcessing:
    """Test signal processing operations"""

    def test_convolve(self):
        signal = ghost.array([1, 2, 3, 4, 5])
        kernel = ghost.array([1, 1, 1])
        result = ghost.convolve(signal, kernel, mode='valid')
        assert np.allclose(np.asarray(result), [6, 9, 12])
        signal.release()
        kernel.release()
        result.release()

    def test_fft_ifft(self):
        a = ghost.array([1.0, 2.0, 3.0, 4.0])
        freq = ghost.fft(a)
        reconstructed = ghost.ifft(freq)
        assert np.allclose(np.real(np.asarray(reconstructed)), np.asarray(a))
        a.release()
        freq.release()
        reconstructed.release()

class TestSecurityValidation:
    """Test security validation"""

    def test_invalid_shape_negative(self):
        with pytest.raises(ValueError):
            ghost.zeros((-10, 5))

    def test_invalid_axis(self):
        a = ghost.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            ghost.sum(a, axis=5)
        a.release()

    def test_shape_overflow_protection(self):
        # Try to create array with dimensions that would overflow
        with pytest.raises(ValueError):
            ghost.zeros((2**30, 2**30))

class TestCheckpointRestart:
    """Test checkpoint and restart functionality"""

    def test_save_load_checkpoint(self, tmp_path):
        checkpoint_file = tmp_path / "test_checkpoint.json"

        # Save checkpoint
        success = ghost.save_checkpoint(str(checkpoint_file))
        assert success
        assert checkpoint_file.exists()

        # Load checkpoint
        success = ghost.load_checkpoint(str(checkpoint_file))
        assert success

    def test_export_profile(self, tmp_path):
        profile_file = tmp_path / "test_profile.json"

        # Run some operations
        a = ghost.array([1, 2, 3])
        b = ghost.add(a, a)
        a.release()
        b.release()

        # Export profile
        success = ghost.export_profile(str(profile_file))
        assert success
        assert profile_file.exists()

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_array(self):
        a = ghost.array([])
        assert a.shape == (0,)
        a.release()

    def test_scalar_to_array(self):
        a = ghost.array(5)
        assert a.shape == ()
        a.release()

    def test_broadcasting(self):
        a = ghost.array([[1, 2, 3]])
        b = ghost.array([[10], [20]])
        c = ghost.add(a, b)
        expected = np.array([[11, 12, 13], [21, 22, 23]])
        assert np.allclose(np.asarray(c), expected)
        a.release()
        b.release()
        c.release()

    def test_abs(self):
        a = ghost.array([-1, -2, 3, -4])
        b = ghost.abs(a)
        assert np.allclose(np.asarray(b), [1, 2, 3, 4])
        a.release()
        b.release()

    def test_sign(self):
        a = ghost.array([-5, 0, 5])
        b = ghost.sign(a)
        assert np.allclose(np.asarray(b), [-1, 0, 1])
        a.release()
        b.release()


class TestInternationalization:
    """Test internationalization system"""

    def test_i18n_import(self):
        """Test i18n module can be imported"""
        try:
            from i18n import get_i18n_manager, translate, set_language
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import i18n module: {e}")

    def test_i18n_manager_creation(self):
        """Test i18n manager creation"""
        from i18n import get_i18n_manager
        manager = get_i18n_manager()
        assert manager is not None
        assert manager.current_language in manager.SUPPORTED_LANGUAGES

    def test_language_switching(self):
        """Test language switching functionality"""
        from i18n import get_i18n_manager
        manager = get_i18n_manager()

        # Test setting different languages
        languages_to_test = ['en', 'ja', 'zh', 'ko', 'es', 'fr', 'de']
        for lang in languages_to_test:
            if lang in manager.SUPPORTED_LANGUAGES:
                success = manager.set_language(lang)
                assert success
                assert manager.current_language == lang

    def test_translation_functionality(self):
        """Test basic translation functionality"""
        from i18n import get_i18n_manager
        manager = get_i18n_manager()

        # Test English translations
        manager.set_language('en')
        assert manager.translate('app.name') == 'GhostGPU'
        assert manager.translate('error.invalid_input', input='test') == 'Invalid input: test'

        # Test Japanese translations
        manager.set_language('ja')
        assert manager.translate('app.name') == 'GhostGPU'
        assert manager.translate('error.invalid_input', input='テスト') == '無効な入力: テスト'

    def test_fallback_translations(self):
        """Test fallback to English when translation not available"""
        from i18n import get_i18n_manager
        manager = get_i18n_manager()

        # Set to a language that might not have complete translations
        manager.set_language('pt')  # Portuguese
        translation = manager.translate('app.name')
        # Should fall back to English or key itself
        assert translation in ['GhostGPU', 'app.name']

    def test_missing_translations_detection(self):
        """Test detection of missing translations"""
        from i18n import get_i18n_manager
        manager = get_i18n_manager()

        # Check English as base
        manager.set_language('en')
        missing_keys = manager.get_missing_translations('ja')
        # Should return keys that are in English but missing in Japanese
        assert isinstance(missing_keys, list)

    def test_locale_directory_creation(self):
        """Test locale directory and files creation"""
        import os
        from pathlib import Path

        locale_dir = Path('locale')
        assert locale_dir.exists()

        # Check that translation files exist
        expected_files = ['en.json', 'ja.json', 'zh.json', 'ko.json', 'es.json', 'fr.json', 'de.json', 'pt.json', 'ru.json']
        for file in expected_files:
            file_path = locale_dir / file
            assert file_path.exists(), f"Translation file {file} not found"

    def test_translation_file_format(self):
        """Test translation file format and content"""
        import json
        from pathlib import Path

        locale_dir = Path('locale')
        for lang_file in locale_dir.glob('*.json'):
            with open(lang_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert len(data) > 0

                # Check that essential keys exist
                essential_keys = ['app.name', 'nav.brand', 'settings.title']
                for key in essential_keys:
                    assert key in data, f"Essential key '{key}' missing in {lang_file.name}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
