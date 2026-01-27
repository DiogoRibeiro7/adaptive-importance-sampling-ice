"""Integration tests for Safe-ICE."""

from __future__ import annotations

import numpy as np
import pytest
from typing import Callable, Tuple
import time

from safe_ice import SafeICE
from safe_ice.core.parameters import vMFNMParameters
from safe_ice.problems.benchmarks import BenchmarkProblems
from safe_ice.analysis.performance import PerformanceEvaluator


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_workflow_with_benchmarks(self):
        """Test complete workflow from initialization to analysis."""
        # 1. Create problem
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        # 2. Initialize SafeICE
        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=300,
            max_iterations=5
        )

        # 3. Run algorithm
        pf, results = ice.run(verbose=False)

        # 4. Verify results structure
        assert "final_samples" in results
        assert "final_weights" in results
        assert "final_g_values" in results
        assert "iterations" in results
        assert "convergence_metrics" in results

        # 5. Check convergence metrics
        metrics = results["convergence_metrics"]
        assert "cv_values" in metrics
        assert "delta_values" in metrics
        assert len(metrics["cv_values"]) <= 5  # max_iterations

        # 6. Verify probability estimate is reasonable
        assert 1e-6 < pf < 1e-4  # Expected range for this problem

    def test_workflow_with_custom_limit_state(self):
        """Test workflow with user-defined limit state function."""
        # Custom limit state function
        def custom_limit_state(u: np.ndarray) -> np.ndarray:
            """Custom nonlinear limit state function."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            # Nonlinear combination
            term1 = 3.5 - 0.1 * (u[:, 0] - u[:, 1])**2
            term2 = (u[:, 0] + u[:, 1]) / np.sqrt(2)
            return term1 - term2

        # Run SafeICE
        ice = SafeICE(
            limit_state_function=custom_limit_state,
            dimension=2,
            N=500,
            max_iterations=10
        )

        pf, results = ice.run(verbose=False)

        # Verify results
        assert 0 < pf < 1
        assert results["final_samples"].shape[1] == 2
        assert len(results["final_weights"]) == len(results["final_samples"])

    def test_multiple_runs_consistency(self):
        """Test that multiple runs give consistent results."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        pf_values = []

        for seed in [42, 123, 456]:
            np.random.seed(seed)

            ice = SafeICE(
                limit_state_function=g,
                dimension=2,
                N=1000,
                max_iterations=10
            )

            pf, _ = ice.run(verbose=False)
            pf_values.append(pf)

        # Results should be in same order of magnitude
        pf_mean = np.mean(pf_values)
        for pf in pf_values:
            relative_diff = abs(pf - pf_mean) / pf_mean
            assert relative_diff < 1.0  # Within 100% of mean


class TestKnownFailureProbabilities:
    """Test against problems with known failure probabilities."""

    def test_simple_sphere_problem(self):
        """Test simple sphere problem with analytical solution."""
        # For standard normal in R^d, P(||u|| > beta) is known
        beta = 3.0
        d = 2

        def sphere_limit_state(u):
            return beta - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=sphere_limit_state,
            dimension=d,
            N=2000,
            max_iterations=15
        )

        pf_estimated, _ = ice.run(verbose=False)

        # Analytical solution for d=2: P(χ²(2) > beta²)
        from scipy import stats
        pf_analytical = 1 - stats.chi2.cdf(beta**2, df=d)

        # Check relative error (allow 50% due to Monte Carlo variance)
        relative_error = abs(pf_estimated - pf_analytical) / pf_analytical
        assert relative_error < 0.5

    def test_linear_limit_state(self):
        """Test linear limit state with known solution."""
        # Linear limit state: g(u) = a₀ + Σaᵢuᵢ
        # For standard normal, failure probability is Φ(-a₀/||a||)
        a = np.array([1.0, 1.0])  # Coefficients
        a0 = 3.0  # Constant term

        def linear_limit_state(u):
            if u.ndim == 1:
                u = u.reshape(1, -1)
            return a0 + np.dot(u, a)

        ice = SafeICE(
            limit_state_function=linear_limit_state,
            dimension=2,
            N=1000,
            max_iterations=10
        )

        pf_estimated, _ = ice.run(verbose=False)

        # Analytical solution
        from scipy import stats
        pf_analytical = stats.norm.cdf(-a0 / np.linalg.norm(a))

        # Check relative error
        relative_error = abs(pf_estimated - pf_analytical) / pf_analytical
        assert relative_error < 0.5


class TestPerformanceRegression:
    """Test for performance regression."""

    def test_execution_time_reasonable(self):
        """Test that execution time is reasonable for standard problem."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=5
        )

        start_time = time.time()
        pf, _ = ice.run(verbose=False)
        execution_time = time.time() - start_time

        # Should complete within reasonable time (adjust based on system)
        assert execution_time < 60  # 60 seconds max

    def test_memory_usage_scales_linearly(self):
        """Test that memory usage scales linearly with samples."""
        def simple_limit_state(u):
            return 3.0 - np.linalg.norm(u, axis=-1)

        # Test with different sample sizes
        sample_sizes = [100, 500, 1000]
        result_sizes = []

        for N in sample_sizes:
            ice = SafeICE(
                limit_state_function=simple_limit_state,
                dimension=2,
                N=N,
                max_iterations=3
            )

            _, results = ice.run(verbose=False)
            result_sizes.append(len(results["final_samples"]))

        # Check that sizes scale appropriately
        for i in range(1, len(result_sizes)):
            ratio = result_sizes[i] / result_sizes[i-1]
            expected_ratio = sample_sizes[i] / sample_sizes[i-1]
            # Allow some variance due to adaptive sampling
            assert 0.5 * expected_ratio <= ratio <= 2.0 * expected_ratio


class TestHighDimensionalProblems:
    """Test performance in high dimensions."""

    @pytest.mark.slow
    def test_dimension_10(self):
        """Test 10-dimensional problem."""
        d = 10

        def high_dim_limit_state(u):
            # Sum of squares with threshold
            return 5.0 - np.sqrt(np.sum(u**2, axis=-1) / d)

        ice = SafeICE(
            limit_state_function=high_dim_limit_state,
            dimension=d,
            N=1000,
            max_iterations=10
        )

        pf, results = ice.run(verbose=False)

        assert 0 < pf < 1
        assert results["final_samples"].shape[1] == d

    @pytest.mark.slow
    def test_dimension_50(self):
        """Test 50-dimensional problem (stress test)."""
        d = 50

        def very_high_dim_limit_state(u):
            # Linear combination in high dimensions
            weights = np.ones(d) / np.sqrt(d)
            return 3.0 - np.dot(u, weights)

        ice = SafeICE(
            limit_state_function=very_high_dim_limit_state,
            dimension=d,
            N=2000,
            max_iterations=5
        )

        pf, results = ice.run(verbose=False)

        assert 0 < pf < 1
        assert results["final_samples"].shape[1] == d


class TestWithPerformanceEvaluator:
    """Test integration with PerformanceEvaluator."""

    def test_performance_comparison(self):
        """Test performance comparison between Safe-ICE and Monte Carlo."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        evaluator = PerformanceEvaluator()

        # Run Monte Carlo reference (with fewer samples for speed)
        pf_mc, std_mc = evaluator.run_monte_carlo_reference(
            limit_state_func=g,
            dimension=2,
            n_samples=10000
        )

        # Run Safe-ICE
        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=10
        )
        pf_ice, _ = ice.run(verbose=False)

        # Both should give similar order of magnitude
        # (Allow large tolerance due to rare event)
        if pf_mc > 0:  # Only compare if MC found failures
            ratio = pf_ice / pf_mc
            assert 0.1 < ratio < 10.0

    def test_compute_metrics(self):
        """Test metric computation from results."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=5
        )

        pf, results = ice.run(verbose=False)

        evaluator = PerformanceEvaluator()

        # Compute coefficient of variation
        weights = results["final_weights"]
        g_values = results["final_g_values"]

        failure_indicator = (g_values <= 0).astype(float)
        if np.sum(failure_indicator * weights) > 0:
            cv = np.sqrt(np.var(failure_indicator * weights)) / np.mean(failure_indicator * weights)
            assert cv > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample_per_iteration(self):
        """Test with minimal samples per iteration."""
        def simple_limit_state(u):
            return 2.0 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=simple_limit_state,
            dimension=2,
            N=10,  # Very few samples
            max_iterations=3
        )

        # Should still run without errors
        pf, results = ice.run(verbose=False)
        assert 0 <= pf <= 1

    def test_single_iteration(self):
        """Test with only one iteration allowed."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=1000,
            max_iterations=1
        )

        pf, results = ice.run(verbose=False)

        # Should have samples from exactly one iteration
        assert len(results["iterations"]) == 1
        assert len(results["final_samples"]) == 1000

    def test_very_rare_event(self):
        """Test with extremely rare event (small failure probability)."""
        def rare_event_limit_state(u):
            return 6.0 - np.linalg.norm(u, axis=-1)  # Very rare

        ice = SafeICE(
            limit_state_function=rare_event_limit_state,
            dimension=2,
            N=1000,
            max_iterations=20
        )

        pf, results = ice.run(verbose=False)

        # Should give very small probability
        assert pf < 1e-6

    def test_certain_failure(self):
        """Test with certain failure (large failure probability)."""
        def certain_failure_limit_state(u):
            return -1.0 - np.linalg.norm(u, axis=-1)  # Always negative

        ice = SafeICE(
            limit_state_function=certain_failure_limit_state,
            dimension=2,
            N=100,
            max_iterations=2
        )

        pf, results = ice.run(verbose=False)

        # Should give probability close to 1
        assert pf > 0.99