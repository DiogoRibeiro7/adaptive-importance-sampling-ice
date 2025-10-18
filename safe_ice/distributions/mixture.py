# safe_ice/distributions/mixture.py
"""von Mises–Fisher–Nakagami mixture distribution."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import iv, gamma  # noqa: F401

from ..core.parameters import vMFNMParameters
from .vmf import VonMisesFisherSampler
from .nakagami import NakagamiDistribution

NDArrayF = npt.NDArray[np.float64]


class vMFNMDistribution:
    """Complete von Mises–Fisher–Nakagami mixture distribution."""

    def __init__(self, params: vMFNMParameters) -> None:
        self.params = params
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        K, d = self.params.K, self.params.d
        assert self.params.pi.shape == (K,)
        assert self.params.m.shape == (K,)
        assert self.params.Omega.shape == (K,)
        assert self.params.mu.shape == (K, d)
        assert self.params.kappa.shape == (K,)

        assert np.allclose(float(np.sum(self.params.pi)), 1.0)
        assert np.all(self.params.pi >= 0)
        assert np.all(self.params.m > 0)
        assert np.all(self.params.Omega > 0)
        assert np.all(self.params.kappa >= 0)

        # Normalize mu_k to unit vectors
        for k in range(K):
            norm = float(np.linalg.norm(self.params.mu[k]))
            if norm > 0.0:
                self.params.mu[k] = self.params.mu[k] / norm

    def pdf(self, x: npt.ArrayLike) -> NDArrayF:
        """Compute mixture PDF at rows of x. Returns (n,) float64 array."""
        X = np.asarray(x, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d = X.shape
        assert d == self.params.d

        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            xi = X[i]
            r = float(np.linalg.norm(xi))
            if r > 1e-12:
                a = xi / r
            else:
                a = np.zeros(d, dtype=np.float64)
                a[0] = 1.0
                r = 1e-12

            mix_val: float = 0.0
            for k in range(self.params.K):
                nak = float(NakagamiDistribution.pdf(r, float(self.params.m[k]), float(self.params.Omega[k])))
                vmf = float(self._vmf_pdf(a, self.params.mu[k], float(self.params.kappa[k])))
                mix_val += float(self.params.pi[k]) * nak * vmf
            out[i] = mix_val
        return out

    def _vmf_pdf(self, x: NDArrayF, mu: NDArrayF, kappa: float) -> float:
        """von Mises–Fisher pdf at unit x (internal helper)."""
        d = int(x.size)
        if kappa <= 0.0:
            return float(1.0 / sphere_surface_area(d))
        v = d / 2.0 - 1.0
        C = float((kappa**v) / (((2.0 * np.pi) ** (d / 2.0)) * iv(v, kappa)))
        return float(C * np.exp(kappa * float(x @ mu)))

    def sample(self, n_samples: int) -> NDArrayF:
        """Sample from the mixture. Returns (n_samples, d)."""
        n, d = int(n_samples), self.params.d
        samples = np.zeros((n, d), dtype=np.float64)

        comp = np.random.choice(self.params.K, size=n, p=self.params.pi)
        for i in range(n):
            k = int(comp[i])
            r_i = float(NakagamiDistribution.sample(float(self.params.m[k]), float(self.params.Omega[k]), 1)[0])
            a_i = VonMisesFisherSampler.sample(self.params.mu[k], float(self.params.kappa[k]), 1)[0]
            samples[i] = r_i * a_i
        return samples

    def log_likelihood(self, x: npt.ArrayLike) -> float:
        pdf_vals = self.pdf(x)
        return float(np.sum(np.log(np.maximum(pdf_vals, 1e-15))))


def sphere_surface_area(d: int) -> float:
    from scipy.special import gamma  # local import
    return float(2.0 * (np.pi ** (d / 2.0)) / gamma(d / 2.0))
