"""Benchmark problems for Safe-ICE testing."""

import numpy as np
from typing import Callable


class BenchmarkProblems:
    """Complete implementation of benchmark problems from the paper"""

    @staticmethod
    def four_mode_series_system(z: float = 0.0) -> Callable[[np.ndarray], float]:
        """Four-mode series system from Section 4.1 (Equation 37)"""

        def limit_state_function(u: np.ndarray) -> float:
            u1, u2 = u[0], u[1]

            g1 = 0.1 * (u1 - u2) ** 2 - (u1 + u2) / np.sqrt(2) + 3
            g2 = 0.1 * (u1 - u2) ** 2 + (u1 + u2) / np.sqrt(2) + 3
            g3 = u1 - u2 + np.sqrt(7 / 2)
            g4 = u2 - u1 + np.sqrt(7 / 2)

            g_min = min(g1, g2, g3, g4)
            return g_min + z

        return limit_state_function

    @staticmethod
    def three_mode_problem(z: float = 3.0) -> Callable[[np.ndarray], float]:
        """Three-mode problem from Section 4.2 (Equation 38)"""

        def limit_state_function(u: np.ndarray) -> float:
            u1, u2 = u[0], u[1]

            g1 = z - 1 - u2 + np.exp(-(u1**2) / 10) + (u1 / 5) ** 4
            g2 = z**2 / 2 - u1 * u2

            return min(g1, g2)

        return limit_state_function

    @staticmethod
    def nonlinear_oscillator_simplified(
        d: int = 10, z: float = 0.05
    ) -> Callable[[np.ndarray], float]:
        """Simplified nonlinear oscillator problem"""

        def limit_state_function(u: np.ndarray) -> float:
            # Simplified model capturing essential dynamics
            # This approximates the complex Bouc-Wen oscillator

            # Parameters from paper
            m = 6e4  # mass
            k = 5e6  # stiffness
            zeta = 0.05  # damping ratio
            xy = 0.04  # yield displacement
            alpha = 0.1  # force partition

            # White noise parameters
            S = 0.005  # intensity
            omega_cut = 15 * np.pi  # cutoff frequency
            dt = 0.01  # time step
            T = 8.0  # final time

            # Frequency discretization
            domega = omega_cut / (d / 2)
            omega = np.arange(1, d // 2 + 1) * domega

            # Force amplitude
            sigma = np.sqrt(2 * S * domega)

            # Construct force time series (simplified)
            force_rms = sigma * np.sqrt(np.sum(u**2))

            # Simplified response calculation
            omega_n = np.sqrt(k / m)
            response_scale = force_rms / (k * (1 - alpha))

            # Approximate maximum displacement
            max_displacement = response_scale * (1 + 0.5 * force_rms / (k * xy))

            return z - max_displacement

        return limit_state_function

    @staticmethod
    def two_mode_opposite_directions(z: float = 5.5) -> Callable[[np.ndarray], float]:
        """Two-mode problem with opposite directions (Equation 43)"""

        def limit_state_function(u: np.ndarray) -> float:
            d = len(u)
            sum_u = np.sum(u)

            g1 = z - sum_u / np.sqrt(d)
            g2 = z + sum_u / np.sqrt(d)

            return min(g1, g2)

        return limit_state_function
