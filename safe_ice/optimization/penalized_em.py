# safe_ice/optimization/penalized_em.py
"""Penalized EM algorithm for automatic component selection."""

from __future__ import annotations

from typing import Tuple

import math
import numpy as np
import numpy.typing as npt
from scipy.special import iv, ive, gamma

from ..core.parameters import vMFNMParameters
from ..distributions.nakagami import NakagamiDistribution
from ..distributions.mixture import vMFNMDistribution

NDArrayF = npt.NDArray[np.float64]


class PenalizedEMOptimizer:
    """Penalized EM algorithm for automatic component selection."""

    def __init__(self, max_em_iterations: int = 100, em_tolerance: float = 1e-6) -> None:
        self.max_em_iterations = int(max_em_iterations)
        self.em_tolerance = float(em_tolerance)

    def fit(
        self,
        data: NDArrayF,
        weights: NDArrayF,
        initial_params: vMFNMParameters,
        beta_init: float = 1.0,
    ) -> Tuple[vMFNMParameters, int]:
        """
        Fit vMFNM mixture using penalized EM.

        Args:
            data: (n, d) sample data.
            weights: (n,) importance weights.
            initial_params: initial vMFNM parameters.
            beta_init: initial penalty parameter.

        Returns:
            (optimized_params, final_K)
        """
        n, d = data.shape
        params = self._copy_parameters(initial_params)
        beta: float = float(beta_init)
        K: int = int(params.K)

        # Precompute data in polar coordinates
        radii: NDArrayF = np.linalg.norm(data, axis=1).astype(np.float64, copy=False)
        directions: NDArrayF = np.zeros_like(data, dtype=np.float64)
        valid_radii = radii > 1e-12
        directions[valid_radii] = data[valid_radii] / radii[valid_radii, np.newaxis]

        # Handle zero vectors
        if np.any(~valid_radii):
            directions[~valid_radii, 0] = 1.0
            radii[~valid_radii] = 1e-12

        prev_log_likelihood: float = float("-inf")

        for _ in range(self.max_em_iterations):
            # E-step: compute responsibilities
            responsibilities: NDArrayF = self._e_step(
                data, radii, directions, params, weights
            )

            # M-step with penalization
            params, K = self._penalized_m_step(
                data, radii, directions, responsibilities, weights, params, beta
            )

            # Update penalty parameter
            beta = self._update_beta(params, beta, K)

            # Check convergence
            current_log_likelihood: float = self._weighted_log_likelihood(
                data, params, weights
            )
            if abs(current_log_likelihood - prev_log_likelihood) < self.em_tolerance:
                break
            prev_log_likelihood = current_log_likelihood

        return params, K

    # -------------------------------------------------------------------------
    # E-step
    # -------------------------------------------------------------------------
    def _e_step(
        self,
        data: NDArrayF,
        radii: NDArrayF,
        directions: NDArrayF,
        params: vMFNMParameters,
        weights: NDArrayF,
    ) -> NDArrayF:
        """E-step: compute posterior responsibilities."""
        n = int(data.shape[0])
        K = int(params.K)
        responsibilities: NDArrayF = np.zeros((n, K), dtype=np.float64)

        for i in range(n):
            r: float = float(radii[i])
            a: NDArrayF = directions[i]

            # Component likelihoods
            comp_like: NDArrayF = np.zeros(K, dtype=np.float64)
            for k in range(K):
                # Many pdfs are typed for arrays; pass 1-D array and extract scalar.
                r_arr: NDArrayF = np.asarray([r], dtype=np.float64)
                nak_pdf_arr: NDArrayF = np.asarray(
                    NakagamiDistribution.pdf(r_arr, float(params.m[k]), float(params.Omega[k])),
                    dtype=np.float64,
                )
                nakagami_pdf: float = float(nak_pdf_arr[0])

                vmf_pdf: float = self._vmf_pdf_single(a, params.mu[k], float(params.kappa[k]))
                comp_like[k] = float(params.pi[k]) * nakagami_pdf * vmf_pdf

            total_like: float = float(np.sum(comp_like))
            if total_like > 1e-15:
                responsibilities[i, :] = comp_like / total_like
            else:
                responsibilities[i, :] = 1.0 / float(K)

        # Weighting by importance weights is done in M-step via weighted_resp
        return responsibilities

    # -------------------------------------------------------------------------
    # M-step (penalized)
    # -------------------------------------------------------------------------
    def _penalized_m_step(
        self,
        data: NDArrayF,
        radii: NDArrayF,
        directions: NDArrayF,
        responsibilities: NDArrayF,
        weights: NDArrayF,
        params: vMFNMParameters,
        beta: float,
    ) -> Tuple[vMFNMParameters, int]:
        """Penalized M-step with automatic component removal."""
        n, d = data.shape
        K = int(params.K)

        # Weighted responsibilities
        weighted_resp: NDArrayF = (responsibilities * weights[:, np.newaxis]).astype(
            np.float64, copy=False
        )

        # Update mixture weights with penalization
        new_pi: NDArrayF = self._update_mixture_weights_penalized(
            weighted_resp, params.pi, beta
        )

        # Remove components with negligible weights
        active_components = new_pi > 1e-4
        K_new: int = int(np.sum(active_components))

        if K_new == 0:
            K_new = 1
            active_components[0] = True

        # Extract active components
        active_indices = np.where(active_components)[0]
        new_pi = new_pi[active_components]
        new_pi = (new_pi / float(np.sum(new_pi))).astype(np.float64, copy=False)

        # Update other parameters for active components
        new_m: NDArrayF = np.zeros(K_new, dtype=np.float64)
        new_Omega: NDArrayF = np.zeros(K_new, dtype=np.float64)
        new_mu: NDArrayF = np.zeros((K_new, d), dtype=np.float64)
        new_kappa: NDArrayF = np.zeros(K_new, dtype=np.float64)

        for idx, k in enumerate(active_indices):
            resp_k: NDArrayF = weighted_resp[:, k]
            sum_resp: float = float(np.sum(resp_k))

            if sum_resp > 1e-15:
                # Update Nakagami parameters
                m_hat, Omega_hat = self._update_nakagami_parameters(radii, resp_k)
                new_m[idx] = float(m_hat)
                new_Omega[idx] = float(Omega_hat)

                # Update von Mises-Fisher parameters
                mu_hat, kappa_hat = self._update_vmf_parameters(directions, resp_k)
                new_mu[idx, :] = mu_hat
                new_kappa[idx] = float(kappa_hat)
            else:
                # Keep previous parameters
                new_m[idx] = float(params.m[k])
                new_Omega[idx] = float(params.Omega[k])
                new_mu[idx, :] = params.mu[k]
                new_kappa[idx] = float(params.kappa[k])

        new_params = vMFNMParameters(
            pi=new_pi, m=new_m, Omega=new_Omega, mu=new_mu, kappa=new_kappa
        )

        return new_params, K_new

    def _update_mixture_weights_penalized(
        self, weighted_resp: NDArrayF, old_pi: NDArrayF, beta: float
    ) -> NDArrayF:
        """Update mixture weights with cross-entropy-like penalty."""
        # Standard EM update
        total_weight: float = float(np.sum(weighted_resp))
        if total_weight <= 0.0:
            # fallback uniform
            K = int(weighted_resp.shape[1])
            return np.full(K, 1.0 / float(K), dtype=np.float64)

        pi_em: NDArrayF = (np.sum(weighted_resp, axis=0) / total_weight).astype(
            np.float64, copy=False
        )

        # Penalty
        # Encourage spread based on current entropy of old_pi
        safe_old = np.maximum(old_pi.astype(np.float64, copy=False), 1e-15)
        entropy_current: float = float(-np.sum(safe_old * np.log(safe_old)))

        K = int(old_pi.shape[0])
        penalty_term: NDArrayF = np.zeros(K, dtype=np.float64)
        # Normalize factor (total_weight/total_weight == 1), kept explicit for clarity
        norm_factor: float = 1.0
        for k in range(K):
            penalty_term[k] = float(beta) * norm_factor * float(old_pi[k]) * (
                float(np.log(max(float(old_pi[k]), 1e-15))) - float(entropy_current)
            )

        new_pi: NDArrayF = (pi_em + penalty_term).astype(np.float64, copy=False)

        # Ensure non-negativity and renormalization
        new_pi = np.maximum(new_pi, 0.0)
        s = float(np.sum(new_pi))
        if s > 0.0:
            new_pi = (new_pi / s).astype(np.float64, copy=False)
        else:
            new_pi = np.full(K, 1.0 / float(K), dtype=np.float64)

        return new_pi

    def _update_nakagami_parameters(
        self, radii: NDArrayF, responsibilities: NDArrayF
    ) -> Tuple[float, float]:
        """Update Nakagami parameters using method of moments."""
        sum_resp: float = float(np.sum(responsibilities))

        if sum_resp < 1e-15:
            return 1.0, 1.0

        # Weighted moments
        mean_r2: float = float(np.sum(responsibilities * (radii ** 2)) / sum_resp)
        mean_r4: float = float(np.sum(responsibilities * (radii ** 4)) / sum_resp)

        if mean_r4 <= mean_r2 ** 2:
            return 1.0, mean_r2

        # Method-of-moments estimators
        m_est: float = float((mean_r2 ** 2) / (mean_r4 - mean_r2 ** 2))
        Omega_est: float = float(mean_r2)

        # Ensure valid parameters
        m_est = max(m_est, 0.5)
        Omega_est = max(Omega_est, 1e-6)

        return float(m_est), float(Omega_est)

    def _update_vmf_parameters(
        self, directions: NDArrayF, responsibilities: NDArrayF
    ) -> Tuple[NDArrayF, float]:
        """Update von Mises-Fisher parameters."""
        d = int(directions.shape[1])
        sum_resp: float = float(np.sum(responsibilities))

        if sum_resp < 1e-15:
            mu = np.zeros(d, dtype=np.float64)
            mu[0] = 1.0
            return mu, 0.0

        # Weighted mean direction
        mean_direction: NDArrayF = (
            np.sum(responsibilities[:, np.newaxis] * directions, axis=0) / sum_resp
        ).astype(np.float64, copy=False)
        R: float = float(np.linalg.norm(mean_direction))

        if R < 1e-12:
            mu = np.zeros(d, dtype=np.float64)
            mu[0] = 1.0
            kappa = 0.0
        else:
            mu = (mean_direction / R).astype(np.float64, copy=False)
            kappa = self._estimate_kappa(float(R), d)

        return mu, float(kappa)

    def _estimate_kappa(self, R: float, d: int) -> float:
        """Estimate concentration parameter kappa from mean resultant length R."""
        R = float(min(max(R, 0.0), 1.0 - 1e-12))
        if R >= 1.0 - 1e-12:
            return 1_000.0  # very concentrated

        if d == 2:
            # Circular case: approximations for kappa(R)
            if R < 0.53:
                return float(2.0 * R + R ** 3 + 5.0 * (R ** 5) / 6.0)
            elif R < 0.85:
                return float(-0.4 + 1.39 * R + 0.43 / (1.0 - R))
            else:
                denom = R ** 3 - 4.0 * R ** 2 + 3.0 * R
                if abs(denom) < 1e-12:
                    return 1_000.0
                return float(1.0 / denom)
        else:
            # Higher dimensions: Banerjee et al. style approximation
            return float(R * (d - R ** 2) / (1.0 - R ** 2))

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _vmf_pdf_single(self, x: NDArrayF, mu: NDArrayF, kappa: float) -> float:
        """von Mises–Fisher PDF for a single point w.r.t. surface area measure."""
        d = int(x.shape[0])

        if float(kappa) == 0.0:
            # Uniform density on S^{d-1}: 1 / surface_area
            surface_area: float = float(
                2.0 * (math.pi ** (d / 2.0)) / float(gamma(d / 2.0))
            )
            return 1.0 / surface_area

        nu: float = float(d / 2.0 - 1.0)

        # Use exponentially-scaled Bessel to avoid overflow for large κ
        ive_val: float = float(ive(nu, kappa))
        if ive_val <= 0.0 or not np.isfinite(ive_val):
            return 0.0

        log_C = (
            nu * float(np.log(kappa))
            - (d / 2.0) * float(np.log(2.0 * math.pi))
            - float(np.log(ive_val))
            - float(kappa)
        )
        dot_val: float = float(np.dot(x, mu))
        log_pdf = log_C + float(kappa) * dot_val

        if not np.isfinite(log_pdf):
            return 0.0
        if log_pdf < -745.0:
            return 0.0
        if log_pdf > 700.0:
            return float(np.exp(700.0))
        return float(np.exp(log_pdf))

    def _update_beta(self, params: vMFNMParameters, beta: float, K: int) -> float:
        """Update penalty parameter beta (simple adaptive heuristic)."""
        # Use mixture spread as a guide; keep bounded in [0, 1].
        min_pi: float = float(np.min(params.pi))
        max_pi: float = float(np.max(params.pi))

        # Dimension based factor (similar spirit to paper’s dependence on d)
        d: int = int(params.mu.shape[1])
        eta: float = float(min(1.0, 0.5 ** (max(0, int(d / 2) - 1))))

        # Heuristic terms
        # Encourage spreading when weights are peaky (max_pi large, min_pi small)
        term1: float = float(
            (1.0 / float(K)) * np.sum(
                np.exp(-eta * float(K) * np.abs(params.pi - float(np.mean(params.pi))))
            )
        )
        term2: float = float((1.0 - max_pi) / max(min_pi, 1e-15))

        new_beta: float = float(min(term1, term2))
        # Blend with previous beta to avoid oscillations
        return float(0.5 * beta + 0.5 * max(0.0, min(1.0, new_beta)))

    def _weighted_log_likelihood(
        self, data: NDArrayF, params: vMFNMParameters, weights: NDArrayF
    ) -> float:
        """Compute weighted log-likelihood."""
        dist = vMFNMDistribution(params)
        pdf_vals: NDArrayF = np.asarray(dist.pdf(data), dtype=np.float64)
        log_pdf: NDArrayF = np.log(np.maximum(pdf_vals, 1e-15)).astype(np.float64, copy=False)
        return float(np.sum(weights * log_pdf))

    def _copy_parameters(self, params: vMFNMParameters) -> vMFNMParameters:
        """Create a deep copy of parameters."""
        return vMFNMParameters(
            pi=params.pi.copy(),
            m=params.m.copy(),
            Omega=params.Omega.copy(),
            mu=params.mu.copy(),
            kappa=params.kappa.copy(),
        )
