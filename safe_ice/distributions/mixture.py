"""von Mises-Fisher-Nakagami mixture distribution."""

import numpy as np
from scipy.special import iv, gamma

from ..core.parameters import vMFNMParameters
from .vmf import VonMisesFisherSampler
from .nakagami import NakagamiDistribution


class vMFNMDistribution:
    """Complete von Mises-Fisher-Nakagami mixture distribution"""

    def __init__(self, params: vMFNMParameters) -> None:
        self.params = params
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate parameter consistency"""
        K, d = self.params.K, self.params.d

        assert len(self.params.pi) == K
        assert len(self.params.m) == K
        assert len(self.params.Omega) == K
        assert self.params.mu.shape == (K, d)
        assert len(self.params.kappa) == K

        assert np.allclose(np.sum(self.params.pi), 1.0)
        assert np.all(self.params.pi >= 0)
        assert np.all(self.params.m > 0)
        assert np.all(self.params.Omega > 0)
        assert np.all(self.params.kappa >= 0)

        # Ensure mu vectors are unit vectors
        for k in range(K):
            mu_norm = np.linalg.norm(self.params.mu[k])
            if mu_norm > 1e-12:
                self.params.mu[k] = self.params.mu[k] / mu_norm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute PDF at points x"""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_points, d = x.shape
        assert d == self.params.d

        pdf_vals = np.zeros(n_points)

        for i in range(n_points):
            r = np.linalg.norm(x[i])

            if r > 1e-12:
                a = x[i] / r
            else:
                # Handle zero vector
                a = np.zeros(d)
                if d > 0:
                    a[0] = 1.0
                r = 1e-12

            # Compute mixture density
            mixture_val = 0.0
            for k in range(self.params.K):
                # Nakagami component
                nakagami_pdf = NakagamiDistribution.pdf(
                    r, self.params.m[k], self.params.Omega[k]
                )

                # von Mises-Fisher component
                vmf_pdf = self._vmf_pdf(a, self.params.mu[k], self.params.kappa[k])

                mixture_val += self.params.pi[k] * nakagami_pdf * vmf_pdf

            pdf_vals[i] = mixture_val

        return pdf_vals

    def _vmf_pdf(self, x: np.ndarray, mu: np.ndarray, kappa: float) -> float:
        """von Mises-Fisher PDF"""
        d = len(x)

        if kappa == 0:
            # Uniform on sphere
            return 1.0 / self._sphere_surface_area(d)

        # Normalization constant
        C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))

        # PDF value
        return C_d * np.exp(kappa * np.dot(x, mu))

    def _sphere_surface_area(self, d: int) -> float:
        """Surface area of unit sphere in d dimensions"""
        return 2 * np.pi ** (d / 2) / gamma(d / 2)

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from vMFNM mixture"""
        samples = np.zeros((n_samples, self.params.d))

        # Sample mixture components
        component_indices = np.random.choice(
            self.params.K, size=n_samples, p=self.params.pi
        )

        for i in range(n_samples):
            k = component_indices[i]

            # Sample radius from Nakagami
            r = NakagamiDistribution.sample(self.params.m[k], self.params.Omega[k])

            # Sample direction from von Mises-Fisher
            a = VonMisesFisherSampler.sample(
                self.params.mu[k], self.params.kappa[k], 1
            )[0]

            samples[i] = r * a

        return samples

    def log_likelihood(self, x: np.ndarray) -> float:
        """Compute log-likelihood of data"""
        pdf_vals = self.pdf(x)
        return np.sum(np.log(np.maximum(pdf_vals, 1e-15)))
