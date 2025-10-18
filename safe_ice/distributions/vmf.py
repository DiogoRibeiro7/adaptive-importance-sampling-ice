# safe_ice/distributions/vmf.py
"""von Mises–Fisher distribution sampler (Wood, 1994)."""

from __future__ import annotations

import math
import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]


class VonMisesFisherSampler:
    """Exact von Mises–Fisher sampler using Wood's algorithm."""

    @staticmethod
    def sample(mu: npt.ArrayLike, kappa: float, n_samples: int = 1) -> NDArrayF:
        """
        Sample from a vMF_d(mu, kappa).

        Args:
            mu: Mean direction as a length-d array (need not be unit; we normalize).
            kappa: Concentration (>= 0).
            n_samples: Number of samples.

        Returns:
            (n_samples, d) array of unit vectors on S^{d-1}.
        """
        mu_arr: NDArrayF = np.asarray(mu, dtype=np.float64).reshape(-1)
        d: int = int(mu_arr.shape[0])

        # Normalize mu (if zero, raise)
        mu_norm: float = float(np.linalg.norm(mu_arr))
        if mu_norm == 0.0:
            raise ValueError("mu must be non-zero.")
        mu_unit: NDArrayF = mu_arr / mu_norm

        # kappa == 0 => uniform on sphere
        if kappa == 0.0:
            samples: NDArrayF = np.asarray(
                np.random.normal(loc=0.0, scale=1.0, size=(n_samples, d)),
                dtype=np.float64,
            )
            # normalize each row
            norms: NDArrayF = np.linalg.norm(samples, axis=1, keepdims=True).astype(
                np.float64, copy=False
            )
            # avoid division by zero (extremely unlikely)
            eps: float = float(np.finfo(np.float64).tiny)
            norms = np.maximum(norms, eps)
            return (samples / norms).astype(np.float64, copy=False)

        # Special case: d == 2 (circular Von Mises)
        if d == 2:
            return VonMisesFisherSampler._sample_circular(mu_unit, float(kappa), n_samples)

        # General case d >= 3
        out: NDArrayF = np.zeros((n_samples, d), dtype=np.float64)
        for i in range(n_samples):
            w: float = VonMisesFisherSampler._sample_w_wood(float(kappa), d)

            # v ~ Unif(S^{d-2}) in R^{d-1}
            v_raw: NDArrayF = np.asarray(
                np.random.normal(loc=0.0, scale=1.0, size=(d - 1,)),
                dtype=np.float64,
            )
            v_norm: float = float(np.linalg.norm(v_raw))
            if v_norm > 0.0:
                v: NDArrayF = (v_raw / v_norm).astype(np.float64, copy=False)
            else:
                # degenerate (probability ~ 0); just use e_1
                v = np.zeros(d - 1, dtype=np.float64)
                v[0] = 1.0

            # Point on S^{d-1} aligned with e_d
            xy: NDArrayF = np.concatenate(
                [(math.sqrt(max(0.0, 1.0 - w * w)) * v).astype(np.float64, copy=False),
                 np.asarray([w], dtype=np.float64)]
            ).astype(np.float64, copy=False)

            # Rotate e_d -> mu via Householder and apply to xy
            out[i, :] = VonMisesFisherSampler._householder_rotation(xy, mu_unit)

        return out

    @staticmethod
    def _sample_circular(mu: NDArrayF, kappa: float, n_samples: int) -> NDArrayF:
        """
        Sample on S^1 (d == 2) with mean direction mu and concentration kappa.

        Returns:
            (n_samples, 2) float64 array of unit vectors.
        """
        if mu.shape[0] != 2:
            raise ValueError("Circular case requires mu with dimension 2.")
        angles: NDArrayF = np.asarray(
            np.random.vonmises(mu=0.0, kappa=kappa, size=n_samples), dtype=np.float64
        )
        mu_angle: float = float(np.arctan2(mu[1], mu[0]))
        angles = (angles + mu_angle).astype(np.float64, copy=False)
        return np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float64, copy=False)

    @staticmethod
    def _sample_w_wood(kappa: float, d: int) -> float:
        """
        Sample the last coordinate w using Wood (1994) rejection sampler.
        Returns:
            A Python float in [-1, 1].
        """
        # Wood parameters
        b: float = (d - 1.0) / (2.0 * kappa + math.sqrt(4.0 * kappa * kappa + (d - 1.0) ** 2))
        x0: float = (1.0 - b) / (1.0 + b)
        c: float = kappa * x0 + (d - 1.0) * math.log(1.0 - x0 * x0)

        a_beta: float = (d - 1.0) / 2.0
        b_beta: float = a_beta

        while True:
            # Draws as Python floats
            z: float = float(np.random.beta(a=a_beta, b=b_beta))
            w: float = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u: float = float(np.random.uniform(0.0, 1.0))

            test_val: float = kappa * w + (d - 1.0) * math.log(1.0 - x0 * w) - c
            if test_val >= math.log(u):
                return w

    @staticmethod
    def _householder_rotation(x: npt.ArrayLike, mu: npt.ArrayLike) -> NDArrayF:
        """
        Apply Householder reflection H that maps e_d to mu, then return H @ x.

        Args:
            x: length-d vector on S^{d-1} aligned to e_d (last axis).
            mu: unit vector of length d.

        Returns:
            Rotated vector (length d) as float64 ndarray.
        """
        x_arr: NDArrayF = np.asarray(x, dtype=np.float64).reshape(-1)
        mu_arr: NDArrayF = np.asarray(mu, dtype=np.float64).reshape(-1)
        d: int = int(mu_arr.shape[0])

        if x_arr.shape[0] != d:
            raise ValueError("x and mu must have the same dimension.")

        e_d: NDArrayF = np.zeros(d, dtype=np.float64)
        e_d[-1] = 1.0

        # If already aligned, return x
        if np.allclose(mu_arr, e_d):
            return x_arr.astype(np.float64, copy=False)

        # Householder vector u = e_d - mu (no need to form H explicitly)
        u: NDArrayF = (e_d - mu_arr).astype(np.float64, copy=False)
        u_norm: float = float(np.linalg.norm(u))
        if u_norm < 1e-15:
            return x_arr.astype(np.float64, copy=False)

        u /= u_norm
        # Hx = x - 2 (u^T x) u
        dot: float = float(np.dot(u, x_arr))
        return (x_arr - 2.0 * dot * u).astype(np.float64, copy=False)
