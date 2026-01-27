"""Tests for the SafeICE algorithm."""

from __future__ import annotations

import numpy as np
import pytest
from typing import Callable

from safe_ice import SafeICE
from safe_ice.core.parameters import vMFNMParameters
from safe_ice.problems.benchmarks import BenchmarkProblems


class TestSafeICEInitialization:
    """Test SafeICE initialization and configuration."""

    def test_basic_initialization(self):
        """Test SafeICE can be initialized with minimal parameters."""
        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100
        )

        assert ice is not None
        assert ice.d == 2
        assert ice.N == 100
        assert ice.max_iterations == 20  # default value
        assert ice.delta_target == 4.0  # default value

    def test_initialization_with_custom_parameters(self):
        """Test SafeICE initialization with custom parameters."""
        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=5,
            N=500,
            max_iterations=50,
            delta_target=2.0,
            delta_star=1.0,
            sigma0=2.0,
            K0=15,
            em_max_iter=50
        )

        assert ice.d == 5
        assert ice.N == 500
        assert ice.max_iterations == 50
        assert ice.delta_target == 2.0
        assert ice.delta_star == 1.0
        assert ice.sigma0 == 2.0
        assert ice.K0 == 15



class TestSafeICEExecution:
    """Test SafeICE algorithm execution."""

    def test_single_iteration(self):
        """Test that a single iteration runs without errors."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=1,
        )

        # Run one iteration
        pf, results = ice.run(verbose=False)

        assert pf > 0
        assert pf < 1
        assert "final_samples" in results
        assert "final_weights" in results
        assert results["final_samples"].shape[1] == 2
        assert len(results["final_weights"]) == len(results["final_samples"])
        assert np.all(results["final_weights"] >= 0)

    def test_multiple_iterations(self):
        """Test multiple iterations with convergence."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=200,
            max_iterations=5,
        )

        pf, results = ice.run(verbose=False)

        assert pf > 0
        assert pf < 1e-3  # Known to be a rare event
        assert results["final_samples"].shape[0] >= 200  # At least one iteration
        assert results["final_samples"].shape[0] <= 200 * 5  # At most max_iterations
        assert results["final_samples"].shape[1] == 2
        assert len(results["final_weights"]) == len(results["final_samples"])

    def test_deterministic_limit_state(self):
        """Test with a deterministic limit state function."""
        def g(u):
            # Simple sphere: failure if ||u|| > 3
            return 3.0 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=10,
        )

        pf, results = ice.run(verbose=False)

        # For a 2D standard normal, P(||u|| > 3) should be small
        assert pf > 0
        assert pf < 0.1  # Should be around 0.01

    def test_high_dimensional_problem(self):
        """Test with higher dimensional problem."""
        def g(u):
            # High-dimensional sphere
            return 4.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=10,
            N=500,
            max_iterations=3,
        )

        pf, results = ice.run(verbose=False)

        assert pf > 0
        assert results["final_samples"].shape[1] == 10
        assert len(results["final_weights"]) == len(results["final_samples"])

    @pytest.mark.parametrize("dimension", [2, 5, 10])
    def test_dimension_consistency(self, dimension):
        """Test that output dimensions are consistent with input."""
        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=dimension,
            N=100,
            max_iterations=2,
        )

        pf, results = ice.run(verbose=False)

        assert results["final_samples"].shape[1] == dimension


class TestSafeICEWithInitialParams:
    """Test SafeICE with initial parameter specification."""

    def test_with_initial_params(self):
        """Test SafeICE with user-provided initial parameters."""
        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        # Create initial parameters
        initial_params = vMFNMParameters(
            K=2,  # Two components
            d=2,  # 2D problem
            pi=np.array([0.6, 0.4]),
            m=np.array([2.0, 2.5]),
            Omega=np.array([1.0, 1.2]),
            mu=np.array([[1.0, 0.0], [0.0, 1.0]]) / np.sqrt(2),
            kappa=np.array([5.0, 3.0])
        )

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=200,
            max_iterations=3,
        )

        pf, results = ice.run(
            initial_params=initial_params,
            verbose=False
        )

        assert pf > 0
        assert results["final_samples"].shape[1] == 2


class TestBenchmarkProblems:
    """Test with benchmark problems."""

    def test_four_mode_series_system(self):
        """Test the four-mode series system benchmark."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=10,
        )

        pf, results = ice.run(verbose=False)

        # Reference probability is approximately 1.22e-5
        assert pf > 1e-6
        assert pf < 1e-4

    def test_three_mode_problem(self):
        """Test the three-mode problem benchmark."""
        problems = BenchmarkProblems()
        g = problems.three_mode_problem()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=300,
            max_iterations=5,
        )

        pf, results = ice.run(verbose=False)

        # Reference probability is approximately 2.3e-3
        assert pf > 1e-4
        assert pf < 1e-2

    @pytest.mark.slow
    def test_two_mode_opposite_directions(self):
        """Test the two-mode opposite directions benchmark (slow test)."""
        problems = BenchmarkProblems()
        # Test with dimension 10
        g = problems.two_mode_opposite_directions(dimension=10)

        ice = SafeICE(
            limit_state_function=g,
            dimension=10,
            N=500,
            max_iterations=5
        )

        pf, results = ice.run(verbose=False)

        assert pf > 0
        assert results["final_samples"].shape[1] == 10


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_limit_state_returns_nan(self):
        """Test handling when limit state function returns NaN."""
        def g(u):
            # Return NaN for some inputs
            result = 3.5 - np.linalg.norm(u, axis=-1)
            result[0] = np.nan
            return result

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=2,
        )

        # Should handle NaN gracefully
        with pytest.warns(RuntimeWarning):
            pf, results = ice.run(verbose=False)

    def test_limit_state_returns_inf(self):
        """Test handling when limit state function returns infinity."""
        def g(u):
            # Return infinity for some inputs
            result = 3.5 - np.linalg.norm(u, axis=-1)
            result[0] = np.inf
            return result

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=2,
        )

        # Should handle infinity gracefully
        pf, results = ice.run(verbose=False)
        assert pf >= 0
        assert pf <= 1

    def test_all_samples_fail(self):
        """Test when all samples are in failure region."""
        def g(u):
            # Always in failure (negative values)
            return -1.0 * np.ones(len(u))

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=2,
        )

        pf, results = ice.run(verbose=False)

        # Should give high probability
        assert pf > 0.9

    def test_no_samples_fail(self):
        """Test when no samples are in failure region."""
        def g(u):
            # Never in failure (always positive)
            return 10.0 * np.ones(len(u))

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=2,
        )

        pf, results = ice.run(verbose=False)

        # Should give very low probability
        assert pf < 1e-10


class TestConvergence:
    """Test convergence behavior."""

    def test_cv_convergence(self):
        """Test that CV convergence criterion works."""
        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=1000,
            max_iterations=20,
            delta_target=0.1,  # Strict convergence
            cv_tolerance=0.01,
        )

        pf, results = ice.run(verbose=False)

        # With strict CV, should converge or reach max_iterations
        assert len(results["final_samples"]) <= 1000 * 20

    def test_early_stopping(self):
        """Test that algorithm stops early when converged."""
        def g(u):
            # Simple problem that should converge quickly
            return 2.0 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=20,  # Allow many iterations
            delta_target=0.5,  # Relaxed convergence
        )

        pf, results = ice.run(verbose=False)

        # Should stop before max_iterations
        assert len(results["final_samples"]) < 500 * 20