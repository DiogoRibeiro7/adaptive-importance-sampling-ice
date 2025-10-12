"""Heat transfer problem with Karhunen-Loève expansion."""

import numpy as np
from typing import Callable


class HeatTransferProblem:
    """Complete heat transfer problem implementation from Section 4.5"""

    def __init__(
        self,
        grid_size: int = 21,
        correlation_length: float = 0.2,
        n_terms: int = 10,
        heat_source: float = 2000.0,
    ) -> None:
        """
        Initialize heat transfer problem

        Args:
            grid_size: discretization grid size
            correlation_length: correlation length for random field
            n_terms: number of KL expansion terms
            heat_source: heat source magnitude
        """
        self.grid_size = grid_size
        self.l = correlation_length
        self.n_terms = n_terms
        self.Q = heat_source

        # Domain parameters
        self.domain = (-0.5, 0.5, -0.5, 0.5)  # (x_min, x_max, y_min, y_max)

        # Generate discretization
        self._setup_discretization()

        # Precompute KL expansion basis
        self._setup_kl_expansion()

    def _setup_discretization(self) -> None:
        """Setup finite element discretization"""
        x = np.linspace(self.domain[0], self.domain[1], self.grid_size)
        y = np.linspace(self.domain[2], self.domain[3], self.grid_size)
        self.X, self.Y = np.meshgrid(x, y)

        # Grid points
        self.grid_points = np.column_stack([self.X.ravel(), self.Y.ravel()])
        self.n_points = len(self.grid_points)

    def _setup_kl_expansion(self) -> None:
        """Setup Karhunen-Loève expansion for lognormal random field"""
        # Exponential covariance function: k(x,x') = exp(-||x-x'||/l)
        distances = np.sqrt(
            np.sum(
                (self.grid_points[:, None, :] - self.grid_points[None, :, :]) ** 2,
                axis=2,
            )
        )

        # Covariance matrix
        C = np.exp(-distances / self.l)

        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(C)

        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        self.eigenvals = eigenvals[idx][: self.n_terms]
        self.eigenvecs = eigenvecs[:, idx][:, : self.n_terms]

        # Normalize eigenvectors
        for i in range(self.n_terms):
            self.eigenvecs[:, i] /= np.sqrt(np.sum(self.eigenvecs[:, i] ** 2))

    def generate_permeability_field(self, xi: np.ndarray) -> np.ndarray:
        """Generate lognormal permeability field from KL expansion"""
        # Mean and std parameters for lognormal field
        mu_kappa = 1.0  # mean permeability
        sigma_kappa = 0.3  # std permeability

        # Lognormal parameters
        a_kappa = np.log(mu_kappa**2 / np.sqrt(mu_kappa**2 + sigma_kappa**2))
        b_kappa = np.sqrt(np.log(1 + sigma_kappa**2 / mu_kappa**2))

        # KL expansion
        f_field = np.sum(
            np.sqrt(self.eigenvals) * self.eigenvecs * xi[: self.n_terms], axis=1
        )

        # Lognormal field
        kappa_field = np.exp(a_kappa + b_kappa * f_field)

        return kappa_field.reshape(self.grid_size, self.grid_size)

    def solve_heat_equation(self, kappa_field: np.ndarray) -> np.ndarray:
        """Solve heat equation using finite differences"""
        # Simple finite difference solver
        h = 1.0 / (self.grid_size - 1)  # grid spacing

        # Initialize temperature field
        T = np.zeros((self.grid_size, self.grid_size))

        # Heat source region A = (0.2, 0.3) × (0.2, 0.3)
        x_indices = np.where((self.X >= 0.2) & (self.X <= 0.3))
        y_indices = np.where((self.Y >= 0.2) & (self.Y <= 0.3))
        source_mask = np.zeros_like(T, dtype=bool)
        source_mask[x_indices[0], x_indices[1]] = True

        # Iterative solver (simplified)
        for iteration in range(1000):
            T_old = T.copy()

            # Interior points
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    # Finite difference approximation
                    laplacian = (
                        kappa_field[i + 1, j] * (T_old[i + 1, j] - T_old[i, j])
                        - kappa_field[i - 1, j] * (T_old[i, j] - T_old[i - 1, j])
                    ) / h**2 + (
                        kappa_field[i, j + 1] * (T_old[i, j + 1] - T_old[i, j])
                        - kappa_field[i, j - 1] * (T_old[i, j] - T_old[i, j - 1])
                    ) / h**2

                    # Add heat source
                    source_term = self.Q if source_mask[i, j] else 0.0

                    T[i, j] = T_old[i, j] + 0.01 * (laplacian + source_term)

            # Boundary conditions
            T[0, :] = 0  # Bottom: Dirichlet
            T[-1, :] = T[-2, :]  # Top: Neumann (zero gradient)
            T[:, 0] = 0  # Left: Dirichlet
            T[:, -1] = 0  # Right: Dirichlet

            # Check convergence
            if np.max(np.abs(T - T_old)) < 1e-6:
                break

        return T

    def create_limit_state_function(
        self, threshold: float = 10.0
    ) -> Callable[[np.ndarray], float]:
        """Create limit state function for heat transfer problem"""

        def limit_state_function(u: np.ndarray) -> float:
            # Generate permeability field
            kappa_field = self.generate_permeability_field(u)

            # Solve heat equation
            T_field = self.solve_heat_equation(kappa_field)

            # Evaluation region B = (-0.3, -0.2) × (-0.3, 0.2)
            x_mask = (self.X >= -0.3) & (self.X <= -0.2)
            y_mask = (self.Y >= -0.3) & (self.Y <= 0.2)
            eval_mask = x_mask & y_mask

            # Average temperature in evaluation region
            T_avg = np.mean(T_field[eval_mask])

            return threshold - T_avg

        return limit_state_function
