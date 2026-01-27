"""Performance evaluation utilities for Safe-ICE."""
from __future__ import annotations

from typing import Any, Tuple, cast, Callable, Optional, Dict
import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]
NDArrayI = npt.NDArray[np.int64]

from ..core.safe_ice import SafeICE


class PerformanceEvaluator:
    """Comprehensive performance evaluation and comparison"""

    @staticmethod
    def run_monte_carlo_reference(
        limit_state_func: Callable, dimension: int, n_samples: int = 1000000
    ) -> Tuple[float, float]:
        """Run Monte Carlo simulation for reference"""
        samples = np.random.multivariate_normal(
            np.zeros(dimension), np.eye(dimension), n_samples
        )
        g_values = np.array([limit_state_func(sample) for sample in samples])

        indicators = (g_values <= 0).astype(float)
        pf_mc = np.mean(indicators)
        pf_std = np.sqrt(pf_mc * (1 - pf_mc) / n_samples)

        return pf_mc, pf_std

    @staticmethod
    def compare_methods(
        limit_state_func: Callable,
        dimension: int,
        reference_pf: Optional[float] = None,
        n_runs: int = 10,
        safe_ice_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Comprehensive method comparison"""
        if safe_ice_params is None:
            safe_ice_params = {}

        print(f"Performance Comparison - {dimension}D Problem")
        print("=" * 50)

        # Safe-ICE results
        safe_ice_results = []
        safe_ice_iterations = []
        safe_ice_components = []

        for run in range(n_runs):
            print(f"Safe-ICE Run {run + 1}/{n_runs}")

            safe_ice = SafeICE(limit_state_func, dimension, **safe_ice_params)
            pf_estimate, results = safe_ice.run(verbose=False)

            safe_ice_results.append(pf_estimate)
            safe_ice_iterations.append(results["iterations"])
            safe_ice_components.append(results["final_components"])

        # Statistics
        safe_ice_mean = np.mean(safe_ice_results)
        safe_ice_std = np.std(safe_ice_results)
        safe_ice_cv = safe_ice_std / safe_ice_mean if safe_ice_mean > 0 else np.inf

        results_dict = {
            "safe_ice": {
                "estimates": safe_ice_results,
                "mean": safe_ice_mean,
                "std": safe_ice_std,
                "cv": safe_ice_cv,
                "mean_iterations": np.mean(safe_ice_iterations),
                "mean_components": np.mean(safe_ice_components),
            }
        }

        # Print results
        print(f"\nSafe-ICE Results ({n_runs} runs):")
        print(f"  Mean Pf: {safe_ice_mean:.6e}")
        print(f"  Std Pf:  {safe_ice_std:.6e}")
        print(f"  CV:      {safe_ice_cv:.4f}")
        print(f"  Avg Iterations: {np.mean(safe_ice_iterations):.1f}")
        print(f"  Avg Components: {np.mean(safe_ice_components):.1f}")

        if reference_pf is not None:
            relative_error = abs(safe_ice_mean - reference_pf) / reference_pf
            print(f"  Relative Error: {relative_error:.4f}")
            results_dict["safe_ice"]["relative_error"] = relative_error

        return results_dict
