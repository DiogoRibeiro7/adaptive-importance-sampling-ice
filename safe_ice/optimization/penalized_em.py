"""Penalized EM algorithm for automatic component selection."""

import numpy as np
from typing import Tuple
from scipy.special import iv, gamma

from ..core.parameters import vMFNMParameters
from ..distributions.nakagami import NakagamiDistribution
from ..distributions.mixture import vMFNMDistribution


class PenalizedEMOptimizer:
    """Penalized EM algorithm for automatic component selection"""

    def __init__(
        self, max_em_iterations: int = 100, em_tolerance: float = 1e-6
    ) -> None:
        self.max_em_iterations = max_em_iterations
        self.em_tolerance = em_tolerance

    def fit(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        initial_params: vMFNMParameters,
        beta_init: float = 1.0,
    ) -> Tuple[vMFNMParameters, int]:
        """
        Fit vMFNM mixture using penalized EM

        Args:
            data: (n, d) sample data
            weights: (n,) importance weights
            initial_params: initial vMFNM parameters
            beta_init: initial penalty parameter

        Returns:
            optimized_params, final_K
        """
        n, d = data.shape
        params = self._copy_parameters(initial_params)
        beta = beta_init
        K = params.K

        # Precompute data in polar coordinates
        radii = np.linalg.norm(data, axis=1)
        directions = np.zeros_like(data)
        valid_radii = radii > 1e-12
        directions[valid_radii] = data[valid_radii] / radii[valid_radii, np.newaxis]

        # Handle zero vectors
        if np.any(~valid_radii):
            directions[~valid_radii, 0] = 1.0
            radii[~valid_radii] = 1e-12

        prev_log_likelihood = -np.inf

        for em_iter in range(self.max_em_iterations):
            # E-step: compute responsibilities
            responsibilities = self._e_step(data, radii, directions, params, weights)

            # M-step with penalization
            params, K = self._penalized_m_step(
                data, radii, directions, responsibilities, weights, params, beta
            )

            # Update penalty parameter
            beta = self._update_beta(params, beta, K)

            # Check convergence
            current_log_likelihood = self._weighted_log_likelihood(
                data, params, weights
            )

            if abs(current_log_likelihood - prev_log_likelihood) < self.em_tolerance:
                break

            prev_log_likelihood = current_log_likelihood

        return params, K

    def _e_step(
        self,
        data: np.ndarray,
        radii: np.ndarray,
        directions: np.ndarray,
        params: vMFNMParameters,
        weights: np.ndarray,
    ) -> np.ndarray:
        """E-step: compute posterior responsibilities"""
        n = len(data)
        K = params.K
        responsibilities = np.zeros((n, K))

        for i in range(n):
            r, a = radii[i], directions[i]

            # Compute component likelihoods
            component_likelihoods = np.zeros(K)
            for k in range(K):
                nakagami_pdf = NakagamiDistribution.pdf(r, params.m[k], params.Omega[k])
                vmf_pdf = self._vmf_pdf_single(a, params.mu[k], params.kappa[k])
                component_likelihoods[k] = params.pi[k] * nakagami_pdf * vmf_pdf

            # Normalize to get responsibilities
            total_likelihood = np.sum(component_likelihoods)
            if total_likelihood > 1e-15:
                responsibilities[i] = component_likelihoods / total_likelihood
            else:
                responsibilities[i] = np.ones(K) / K

        return responsibilities

    def _penalized_m_step(
        self,
        data: np.ndarray,
        radii: np.ndarray,
        directions: np.ndarray,
        responsibilities: np.ndarray,
        weights: np.ndarray,
        params: vMFNMParameters,
        beta: float,
    ) -> Tuple[vMFNMParameters, int]:
        """Penalized M-step with automatic component removal"""
        n, d = data.shape
        K = params.K

        # Weighted responsibilities
        weighted_resp = responsibilities * weights[:, np.newaxis]

        # Update mixture weights with penalization
        new_pi = self._update_mixture_weights_penalized(weighted_resp, params.pi, beta)

        # Remove components with negligible weights
        active_components = new_pi > 1e-4
        K_new = np.sum(active_components)

        if K_new == 0:
            K_new = 1
            active_components[0] = True

        # Extract active components
        active_indices = np.where(active_components)[0]
        new_pi = new_pi[active_components]
        new_pi = new_pi / np.sum(new_pi)  # Renormalize

        # Update other parameters for active components
        new_m = np.zeros(K_new)
        new_Omega = np.zeros(K_new)
        new_mu = np.zeros((K_new, d))
        new_kappa = np.zeros(K_new)

        for idx, k in enumerate(active_indices):
            resp_k = weighted_resp[:, k]
            sum_resp = np.sum(resp_k)

            if sum_resp > 1e-15:
                # Update Nakagami parameters
                new_m[idx], new_Omega[idx] = self._update_nakagami_parameters(
                    radii, resp_k
                )

                # Update von Mises-Fisher parameters
                new_mu[idx], new_kappa[idx] = self._update_vmf_parameters(
                    directions, resp_k
                )
            else:
                # Keep previous parameters
                new_m[idx] = params.m[k]
                new_Omega[idx] = params.Omega[k]
                new_mu[idx] = params.mu[k]
                new_kappa[idx] = params.kappa[k]

        new_params = vMFNMParameters(
            pi=new_pi, m=new_m, Omega=new_Omega, mu=new_mu, kappa=new_kappa
        )

        return new_params, K_new

    def _update_mixture_weights_penalized(
        self, weighted_resp: np.ndarray, old_pi: np.ndarray, beta: float
    ) -> np.ndarray:
        """Update mixture weights with cross-entropy penalty"""
        n, K = weighted_resp.shape

        # Standard EM update
        pi_em = np.sum(weighted_resp, axis=0) / np.sum(weighted_resp)

        # Cross-entropy penalty term
        total_weight = np.sum(weighted_resp)
        entropy_current = -np.sum(old_pi * np.log(np.maximum(old_pi, 1e-15)))

        penalty_term = np.zeros(K)
        for k in range(K):
            penalty_term[k] = (
                beta
                * (total_weight / np.sum(weighted_resp))
                * old_pi[k]
                * (np.log(np.maximum(old_pi[k], 1e-15)) - entropy_current)
            )

        # Penalized update
        new_pi = pi_em + penalty_term

        # Ensure non-negativity and normalization
        new_pi = np.maximum(new_pi, 0)
        if np.sum(new_pi) > 0:
            new_pi = new_pi / np.sum(new_pi)
        else:
            new_pi = np.ones(K) / K

        return new_pi

    def _update_nakagami_parameters(
        self, radii: np.ndarray, responsibilities: np.ndarray
    ) -> Tuple[float, float]:
        """Update Nakagami parameters using method of moments"""
        sum_resp = np.sum(responsibilities)

        if sum_resp < 1e-15:
            return 1.0, 1.0

        # Weighted moments
        mean_r2 = np.sum(responsibilities * radii**2) / sum_resp
        mean_r4 = np.sum(responsibilities * radii**4) / sum_resp

        if mean_r4 <= mean_r2**2:
            return 1.0, mean_r2

        # Method of moments estimators
        m_est = mean_r2**2 / (mean_r4 - mean_r2**2)
        Omega_est = mean_r2

        # Ensure valid parameters
        m_est = max(m_est, 0.5)
        Omega_est = max(Omega_est, 1e-6)

        return m_est, Omega_est

    def _update_vmf_parameters(
        self, directions: np.ndarray, responsibilities: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Update von Mises-Fisher parameters"""
        d = directions.shape[1]
        sum_resp = np.sum(responsibilities)

        if sum_resp < 1e-15:
            mu = np.zeros(d)
            mu[0] = 1.0
            return mu, 0.0

        # Weighted mean direction
        mean_direction = (
            np.sum(responsibilities[:, np.newaxis] * directions, axis=0) / sum_resp
        )
        R = np.linalg.norm(mean_direction)

        if R < 1e-12:
            mu = np.zeros(d)
            mu[0] = 1.0
            kappa = 0.0
        else:
            mu = mean_direction / R
            kappa = self._estimate_kappa(R, d)

        return mu, kappa

    def _estimate_kappa(self, R: float, d: int) -> float:
        """Estimate concentration parameter kappa from mean resultant length R"""
        if R >= 1.0:
            return 1000.0  # Very concentrated

        if d == 2:
            # Circular case: exact formula
            if R < 0.53:
                return 2 * R + R**3 + 5 * R**5 / 6
            elif R < 0.85:
                return -0.4 + 1.39 * R + 0.43 / (1 - R)
            else:
                return 1 / (R**3 - 4 * R**2 + 3 * R)
        else:
            # Higher dimensions: use Banerjee et al. approximation
            return R * (d - R**2) / (1 - R**2)

    def _vmf_pdf_single(self, x: np.ndarray, mu: np.ndarray, kappa: float) -> float:
        """von Mises-Fisher PDF for single point"""
        d = len(x)

        if kappa == 0:
            return 1.0 / (2 * np.pi ** (d / 2) / gamma(d / 2))

        C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))
        return C_d * np.exp(kappa * np.dot(x, mu))

    def _update_beta(self, params: vMFNMParameters, beta: float, K: int) -> float:
        """Update penalty parameter beta"""
        # Adaptive beta update from the paper
        min_pi = np.min(params.pi)
        max_pi = np.max(params.pi)

        # Formula from equation (23) in the paper
        eta = min(1, 0.5 ** (int(params.d / 2) - 1))

        term1 = (1 / K) * np.sum(
            np.exp(-eta * len(params.pi) * np.abs(params.pi - np.mean(params.pi)))
        )
        term2 = (1 - max_pi) / (min_pi + 1e-15)

        return min(term1, term2)

    def _weighted_log_likelihood(
        self, data: np.ndarray, params: vMFNMParameters, weights: np.ndarray
    ) -> float:
        """Compute weighted log-likelihood"""
        dist = vMFNMDistribution(params)
        pdf_vals = dist.pdf(data)
        log_pdf = np.log(np.maximum(pdf_vals, 1e-15))
        return np.sum(weights * log_pdf)

    def _copy_parameters(self, params: vMFNMParameters) -> vMFNMParameters:
        """Create a deep copy of parameters"""
        return vMFNMParameters(
            pi=params.pi.copy(),
            m=params.m.copy(),
            Omega=params.Omega.copy(),
            mu=params.mu.copy(),
            kappa=params.kappa.copy(),
        )
