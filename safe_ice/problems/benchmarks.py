# safe_ice/problems/benchmarks.py
"""Benchmark problems for Safe-ICE testing."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


class BenchmarkProblems:
    """Complete implementation of benchmark problems from the paper."""

    @staticmethod
    def four_mode_series_system(
        z: float = 0.0,
    ) -> Callable[[npt.ArrayLike], float]:
        """Four-mode series system from Section 4.1 (Equation 37)."""

        def limit_state_function(u: npt.ArrayLike) -> float:
            arr = np.asarray(u, dtype=np.float64)
            u1 = float(arr[0])
            u2 = float(arr[1])

            g1 = float(0.1 * (u1 - u2) ** 2 - (u1 + u2) / np.sqrt(2.0) + 3.0)
            g2 = float(0.1 * (u1 - u2) ** 2 + (u1 + u2) / np.sqrt(2.0) + 3.0)
            g3 = float(u1 - u2 + np.sqrt(3.5))  # sqrt(7/2) = sqrt(3.5)
            g4 = float(u2 - u1 + np.sqrt(3.5))

            g_min = min(g1, g2, g3, g4)
            return float(g_min + z)

        return limit_state_function

    @staticmethod
    def three_mode_problem(
        z: float = 3.0,
    ) -> Callable[[npt.ArrayLike], float]:
        """Three-mode problem from Section 4.2 (Equation 38)."""

        def limit_state_function(u: npt.ArrayLike) -> float:
            arr = np.asarray(u, dtype=np.float64)
            u1 = float(arr[0])
            u2 = float(arr[1])

            g1 = float(z - 1.0 - u2 + np.exp(-(u1**2) / 10.0) + (u1 / 5.0) ** 4)
            g2 = float((z**2) / 2.0 - u1 * u2)

            return float(min(g1, g2))

        return limit_state_function

    @staticmethod
    def nonlinear_oscillator_simplified(
        d: int = 10, z: float = 0.05
    ) -> Callable[[npt.ArrayLike], float]:
        """Simplified nonlinear oscillator problem."""

        def limit_state_function(u: npt.ArrayLike) -> float:
            # Parameters from the paper (kept as floats)
            m = 6e4        # mass
            k = 5e6        # stiffness
            alpha = 0.1    # force partition
            S = 0.005      # white-noise intensity

            # Frequency discretization
            d_eff = max(int(d), 2)
            omega_cut = 15.0 * float(np.pi)
            domega = omega_cut / (d_eff / 2.0)

            # Force amplitude
            sigma = float(np.sqrt(2.0 * S * domega))

            # Construct force “RMS” proxy from u
            arr = np.asarray(u, dtype=np.float64)
            force_rms = float(sigma * np.sqrt(float(np.sum(arr**2))))

            # Simplified response calculation
            response_scale = float(force_rms / (k * (1.0 - alpha)))

            # Approximate maximum displacement
            max_displacement = float(
                response_scale * (1.0 + 0.5 * force_rms / (k * 0.04))
            )

            return float(z - max_displacement)

        return limit_state_function

    @staticmethod
    def two_mode_opposite_directions(
        z: float = 5.5,
    ) -> Callable[[npt.ArrayLike], float]:
        """Two-mode problem with opposite directions (Equation 43)."""

        def limit_state_function(u: npt.ArrayLike) -> float:
            arr = np.asarray(u, dtype=np.float64)
            d = int(arr.size)
            sum_u = float(np.sum(arr))

            g1 = float(z - sum_u / np.sqrt(float(d)))
            g2 = float(z + sum_u / np.sqrt(float(d)))

            return float(min(g1, g2))

        return limit_state_function
