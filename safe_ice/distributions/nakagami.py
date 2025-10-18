# safe_ice/distributions/nakagami.py
"""Nakagami and Inverse Nakagami distributions with scalar/array-friendly APIs.

Typing policy
-------------
- Public functions accept either scalars or arrays (ArrayLike).
- If the input is a scalar, return a Python `float`.
- If the input is an array, return `NDArray[np.float64]`.

We expose precise typing with @overload so mypy understands both cases.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt
from scipy.special import gamma, gammainc  # normalized lower incomplete gamma

NDArrayF = npt.NDArray[np.float64]


def _as_float(x: np.ndarray) -> float:
    """Extract a Python float from a 0-d/1-elem array safely."""
    return float(np.asarray(x, dtype=np.float64).reshape(1)[0])


# =============================================================================
# Nakagami(m, Omega) on r > 0
# pdf(r) = 2 * (m^m / (Gamma(m) * Omega^m)) * r^{2m-1} * exp(-m r^2 / Omega), r >= 0
# CDF F(r) = P(m, m r^2 / Omega) where P is the regularized lower incomplete gamma
# Sampling: if X ~ Gamma(shape=m, scale=Omega/m), then R = sqrt(X) ~ Nakagami(m, Omega).
# =============================================================================
class NakagamiDistribution:
    """Nakagami(m, Omega) distribution utilities."""

    # -------------------- PDF --------------------
    @overload
    @staticmethod
    def pdf(r: float, m: float, Omega: float) -> float: ...
    @overload
    @staticmethod
    def pdf(r: NDArrayF, m: float, Omega: float) -> NDArrayF: ...

    @staticmethod
    def pdf(r: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        r_arr: NDArrayF = np.asarray(r, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        # Constants
        coef = 2.0 * ((m ** m) / (float(gamma(m)) * (Omega ** m)))
        x = r_arr

        # Compute pdf on array; enforce r >= 0
        out: NDArrayF = np.zeros_like(x, dtype=np.float64)
        mask = x >= 0.0
        xm = x[mask]
        out[mask] = coef * (xm ** (2.0 * m - 1.0)) * np.exp(-m * (xm ** 2) / Omega)

        # Return scalar if input was scalar
        if np.isscalar(r) or np.asarray(r).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- CDF --------------------
    @overload
    @staticmethod
    def cdf(r: float, m: float, Omega: float) -> float: ...
    @overload
    @staticmethod
    def cdf(r: NDArrayF, m: float, Omega: float) -> NDArrayF: ...

    @staticmethod
    def cdf(r: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        r_arr: NDArrayF = np.asarray(r, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        # For r >= 0: F(r) = P(m, m r^2 / Omega); for r < 0 => 0
        out: NDArrayF = np.zeros_like(r_arr, dtype=np.float64)
        mask = r_arr >= 0.0
        z = (m * (r_arr[mask] ** 2)) / Omega
        out[mask] = gammainc(m, z)  # regularized lower incomplete gamma

        if np.isscalar(r) or np.asarray(r).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- Sample --------------------
    @staticmethod
    def sample(m: float, Omega: float, n: int = 1) -> NDArrayF:
        """Draw `n` Nakagami samples as sqrt of a Gamma(m, Omega/m)."""
        m = float(m)
        Omega = float(Omega)
        n = int(n)
        # X ~ Gamma(shape=m, scale=Omega/m)  (NumPy parameterizes by shape & scale)
        x = np.random.gamma(shape=m, scale=(Omega / m), size=n).astype(np.float64, copy=False)
        return np.sqrt(x, dtype=np.float64)


# =============================================================================
# Inverse Nakagami via transformation: if R ~ Nakagami(m, Omega), define Y = 1 / R.
# Then:
#   pdf_Y(y) = pdf_R(1/y) * (1 / y^2),  y > 0
#   CDF_Y(y) = P(Y <= y) = P(1 / R <= y) = P(R >= 1 / y) = 1 - F_R(1 / y)
# Sampling: draw R ~ Nakagami(m, Omega), return 1 / R.
# This produces a heavy-tailed distribution as desired in Safe-ICE.
# =============================================================================
class InverseNakagamiDistribution:
    """Inverse Nakagami distribution defined by Y = 1 / R, R ~ Nakagami(m, Omega)."""

    # -------------------- PDF --------------------
    @overload
    @staticmethod
    def pdf(y: float, m: float, Omega: float) -> float: ...
    @overload
    @staticmethod
    def pdf(y: NDArrayF, m: float, Omega: float) -> NDArrayF: ...

    @staticmethod
    def pdf(y: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        y_arr: NDArrayF = np.asarray(y, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        out: NDArrayF = np.zeros_like(y_arr, dtype=np.float64)
        mask = y_arr > 0.0
        ym = y_arr[mask]
        # pdf_Y(y) = f_R(1/y) * (1 / y^2)
        r_inv = 1.0 / ym
        fr = NakagamiDistribution.pdf(r_inv, m, Omega)  # returns ndarray here
        out[mask] = fr * (1.0 / (ym ** 2))

        if np.isscalar(y) or np.asarray(y).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- CDF --------------------
    @overload
    @staticmethod
    def cdf(y: float, m: float, Omega: float) -> float: ...
    @overload
    @staticmethod
    def cdf(y: NDArrayF, m: float, Omega: float) -> NDArrayF: ...

    @staticmethod
    def cdf(y: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        y_arr: NDArrayF = np.asarray(y, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        out: NDArrayF = np.zeros_like(y_arr, dtype=np.float64)
        # For y <= 0, CDF = 0. For y > 0: P(Y <= y) = P(R >= 1/y) = 1 - F_R(1/y).
        mask = y_arr > 0.0
        ym = y_arr[mask]
        r_thresh = 1.0 / ym
        Fr = NakagamiDistribution.cdf(r_thresh, m, Omega)  # ndarray
        out[mask] = 1.0 - Fr

        if np.isscalar(y) or np.asarray(y).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- Sample --------------------
    @staticmethod
    def sample(m: float, Omega: float, n: int = 1) -> NDArrayF:
        """Draw `n` samples via inverse transform Y = 1 / R, R ~ Nakagami(m, Omega)."""
        r = NakagamiDistribution.sample(m, Omega, n)
        # Avoid division by zero (practically not needed, r>0 a.s.; still guard)
        eps = np.finfo(np.float64).tiny
        return 1.0 / np.maximum(r, eps)
