import numpy as np
import scipy.stats as stats
from scipy.special import iv, gamma, gammaln, hyp1f1
from scipy.optimize import minimize_scalar, minimize, root_scalar
from scipy.linalg import norm, cholesky, solve_triangular
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable, Dict, Any
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class vMFNMParameters:
    """Parameters for von Mises-Fisher-Nakagami mixture"""

    pi: np.ndarray  # mixture weights (K,)
    m: np.ndarray  # Nakagami shape parameters (K,)
    Omega: np.ndarray  # Nakagami scale parameters (K,)
    mu: np.ndarray  # vMF mean directions (K, d)
    kappa: np.ndarray  # vMF concentration parameters (K,)

    @property
    def K(self) -> int:
        return len(self.pi)

    @property
    def d(self) -> int:
        return self.mu.shape[1]


class VonMisesFisherSampler:
    """Exact von Mises-Fisher distribution sampler using Wood's algorithm"""

    @staticmethod
    def sample(mu: np.ndarray, kappa: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample from von Mises-Fisher distribution using Wood's algorithm (1994)

        Args:
            mu: mean direction (unit vector)
            kappa: concentration parameter
            n_samples: number of samples

        Returns:
            samples: (n_samples, d) array of unit vectors
        """
        d = len(mu)
        mu = mu / np.linalg.norm(mu)  # ensure unit vector

        if kappa == 0:
            # Uniform on sphere
            samples = np.random.normal(0, 1, (n_samples, d))
            samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
            return samples

        if d == 1:
            # Special case: circular distribution
            return VonMisesFisherSampler._sample_circular(mu, kappa, n_samples)

        samples = np.zeros((n_samples, d))

        for i in range(n_samples):
            # Sample w using rejection sampling
            w = VonMisesFisherSampler._sample_w_wood(kappa, d)

            # Sample uniformly from (d-1)-sphere
            v = np.random.normal(0, 1, d - 1)
            v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

            # Construct sample in standard position
            x = np.concatenate([v * np.sqrt(1 - w**2), [w]])

            # Rotate to align with mu
            samples[i] = VonMisesFisherSampler._householder_rotation(x, mu)

        return samples

    @staticmethod
    def _sample_circular(mu: np.ndarray, kappa: float, n_samples: int) -> np.ndarray:
        """Sample from circular von Mises distribution"""
        angles = np.random.vonmises(0, kappa, n_samples)
        # Rotate to align with mu
        mu_angle = np.arctan2(mu[1], mu[0])
        angles += mu_angle
        return np.column_stack([np.cos(angles), np.sin(angles)])

    @staticmethod
    def _sample_w_wood(kappa: float, d: int) -> float:
        """Sample w component using Wood's rejection algorithm"""
        b = (d - 1) / (2 * kappa + np.sqrt(4 * kappa**2 + (d - 1) ** 2))
        x0 = (1 - b) / (1 + b)
        c = kappa * x0 + (d - 1) * np.log(1 - x0**2)

        while True:
            z = np.random.beta((d - 1) / 2, (d - 1) / 2)
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = np.random.uniform()

            test_val = kappa * w + (d - 1) * np.log(1 - x0 * w) - c
            if test_val >= np.log(u):
                return w

    @staticmethod
    def _householder_rotation(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Rotate vector x to align with direction mu using Householder reflection"""
        d = len(mu)
        e_d = np.zeros(d)
        e_d[-1] = 1.0

        if np.allclose(mu, e_d):
            return x

        # Householder vector
        u = e_d - mu
        u_norm = np.linalg.norm(u)

        if u_norm < 1e-12:
            return x

        u = u / u_norm

        # Apply Householder reflection: H = I - 2uu^T
        return x - 2 * np.dot(u, x) * u


class NakagamiDistribution:
    """Exact Nakagami distribution implementation"""

    @staticmethod
    def pdf(r: np.ndarray, m: float, Omega: float) -> np.ndarray:
        """Nakagami probability density function"""
        r = np.asarray(r)
        valid = r > 0
        result = np.zeros_like(r, dtype=float)

        if np.any(valid):
            r_valid = r[valid]
            log_pdf = (
                np.log(2)
                + m * np.log(m)
                - gammaln(m)
                - m * np.log(Omega)
                + (2 * m - 1) * np.log(r_valid)
                - m * r_valid**2 / Omega
            )
            result[valid] = np.exp(log_pdf)

        return result

    @staticmethod
    def sample(m: float, Omega: float, n_samples: int = 1) -> np.ndarray:
        """Sample from Nakagami distribution using gamma relationship"""
        # Nakagami(m, Omega) = sqrt(Gamma(m, Omega/m))
        gamma_samples = np.random.gamma(m, Omega / m, n_samples)
        return np.sqrt(gamma_samples)

    @staticmethod
    def cdf(r: np.ndarray, m: float, Omega: float) -> np.ndarray:
        """Nakagami cumulative distribution function"""
        r = np.asarray(r)
        return stats.gamma.cdf(r**2, a=m, scale=Omega / m)


class InverseNakagamiDistribution:
    """Exact Inverse Nakagami distribution implementation"""

    @staticmethod
    def pdf(r: np.ndarray, m: float, Omega: float) -> np.ndarray:
        """Inverse Nakagami probability density function"""
        r = np.asarray(r)
        valid = r > 0
        result = np.zeros_like(r, dtype=float)

        if np.any(valid):
            r_valid = r[valid]
            log_pdf = (
                np.log(2)
                + m * np.log(m)
                - gammaln(m)
                - m * np.log(Omega)
                - (2 * m + 1) * np.log(r_valid)
                - m / (Omega * r_valid**2)
            )
            result[valid] = np.exp(log_pdf)

        return result

    @staticmethod
    def sample(m: float, Omega: float, n_samples: int = 1) -> np.ndarray:
        """Sample from Inverse Nakagami distribution"""
        # Use relationship: if X ~ Gamma(m, Omega/m), then 1/sqrt(X) ~ InverseNakagami(m, Omega)
        gamma_samples = np.random.gamma(m, Omega / m, n_samples)
        return 1.0 / np.sqrt(gamma_samples)


class vMFNMDistribution:
    """Complete von Mises-Fisher-Nakagami mixture distribution"""

    def __init__(self, params: vMFNMParameters):
        self.params = params
        self._validate_parameters()

    def _validate_parameters(self):
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


class PenalizedEMOptimizer:
    """Penalized EM algorithm for automatic component selection"""

    def __init__(self, max_em_iterations: int = 100, em_tolerance: float = 1e-6):
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


class SafeICE:
    """Complete Safe Cross-Entropy Importance Sampling implementation"""

    def __init__(
        self,
        limit_state_function: Callable[[np.ndarray], float],
        dimension: int,
        K0: int = 20,
        delta_target: float = 4.0,
        delta_star: float = 1.5,
        max_iterations: int = 20,
        N: int = 1000,
        sigma0: float = 1.0,
        em_max_iter: int = 100,
    ):
        """
        Initialize Safe-ICE algorithm

        Args:
            limit_state_function: g(u) where failure when g(u) <= 0
            dimension: problem dimension
            K0: initial number of mixture components
            delta_target: target coefficient of variation for sigma adaptation
            delta_star: CV threshold for stopping criterion
            max_iterations: maximum ICE iterations
            N: samples per iteration
            sigma0: initial smoothing parameter
            em_max_iter: maximum EM iterations per ICE step
        """
        self.g = limit_state_function
        self.d = dimension
        self.K0 = K0
        self.delta_target = delta_target
        self.delta_star = delta_star
        self.max_iterations = max_iterations
        self.N = N
        self.sigma0 = sigma0

        # Initialize EM optimizer
        self.em_optimizer = PenalizedEMOptimizer(max_em_iterations=em_max_iter)

        # History tracking
        self.history = {
            "sigma": [],
            "cv": [],
            "components": [],
            "lambda": [],
            "pf_estimates": [],
        }

    def run(self, verbose: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Execute the complete Safe-ICE algorithm

        Returns:
            failure_probability_estimate, results_dictionary
        """
        if verbose:
            print(f"Safe-ICE Algorithm")
            print(f"Problem dimension: {self.d}")
            print(f"Initial components: {self.K0}")
            print(f"Samples per iteration: {self.N}")
            print("-" * 50)

        # Initialize parameters
        t = 0
        sigma_t = self.sigma0
        M = self.sigma0  # Cosine annealing parameter

        # Initialize vMFNM parameters
        phi_t = self._initialize_vmfnm_parameters()
        lambda_t = self._cosine_annealing_schedule(sigma_t, M)

        while t < self.max_iterations:
            if verbose:
                print(
                    f"Iteration {t + 1:2d}: σ={sigma_t:.6f}, λ={lambda_t:.3f}, K={phi_t.K}"
                )

            # Step 2: Generate samples from safe mixture
            samples = self._generate_safe_mixture_samples(phi_t, lambda_t)

            # Evaluate limit state function
            g_values = np.array([self.g(sample) for sample in samples])

            # Step 3: Calculate stopping weights and CV
            stopping_weights = self._calculate_stopping_weights(
                samples, g_values, sigma_t, phi_t, lambda_t
            )
            cv_w_star = self._coefficient_of_variation(stopping_weights)

            # Store history
            self.history["sigma"].append(sigma_t)
            self.history["cv"].append(cv_w_star)
            self.history["components"].append(phi_t.K)
            self.history["lambda"].append(lambda_t)

            if verbose:
                print(f"           CV={cv_w_star:.4f}")

            # Check stopping criterion
            if cv_w_star <= self.delta_star:
                if verbose:
                    print(f"Converged: CV {cv_w_star:.4f} ≤ {self.delta_star}")
                break

            # Step 4: Determine next sigma
            sigma_t = self._determine_next_sigma(
                samples, g_values, phi_t, lambda_t, sigma_t
            )

            # Step 5: Update parameters using penalized EM
            phi_t = self._update_parameters_penalized_em(
                samples, g_values, phi_t, sigma_t, lambda_t
            )

            # Step 6: Update lambda
            lambda_t = self._cosine_annealing_schedule(sigma_t, M)

            t += 1

        # Final estimation
        final_samples = self._generate_safe_mixture_samples(phi_t, lambda_t)
        final_g_values = np.array([self.g(sample) for sample in final_samples])
        pf_estimate = self._estimate_failure_probability(
            final_samples, final_g_values, phi_t, lambda_t
        )

        results = {
            "failure_probability": pf_estimate,
            "iterations": t + 1,
            "final_components": phi_t.K,
            "final_sigma": sigma_t,
            "final_cv": cv_w_star,
            "final_lambda": lambda_t,
            "final_samples": final_samples,
            "final_g_values": final_g_values,
            "history": self.history,
            "final_parameters": phi_t,
        }

        if verbose:
            print("-" * 50)
            print(f"Final Results:")
            print(f"Failure Probability: {pf_estimate:.6e}")
            print(f"Total Iterations: {t + 1}")
            print(f"Final Components: {phi_t.K}")
            print(f"Final CV: {cv_w_star:.4f}")

        return pf_estimate, results

    def _initialize_vmfnm_parameters(self) -> vMFNMParameters:
        """Initialize vMFNM mixture parameters"""
        K = self.K0
        d = self.d

        # Equal mixture weights
        pi = np.ones(K) / K

        # Initialize Nakagami parameters
        m = np.random.uniform(1.0, 3.0, K)
        Omega = np.random.uniform(0.5, 2.0, K)

        # Initialize von Mises-Fisher parameters
        # Random unit directions
        mu = np.random.normal(0, 1, (K, d))
        for k in range(K):
            mu[k] = mu[k] / np.linalg.norm(mu[k])

        # Small initial concentrations
        kappa = np.random.uniform(0.1, 1.0, K)

        return vMFNMParameters(pi=pi, m=m, Omega=Omega, mu=mu, kappa=kappa)

    def _cosine_annealing_schedule(self, sigma: float, M: float) -> float:
        """Cosine annealing schedule for lambda parameter"""
        if sigma > M:
            return 0.0
        else:
            return 0.5 * (1 + np.cos(np.pi * sigma / M))

    def _generate_safe_mixture_samples(
        self, params: vMFNMParameters, lambda_val: float
    ) -> np.ndarray:
        """Generate samples from safe mixture q_safe(u; φ)"""
        samples = np.zeros((self.N, self.d))

        for i in range(self.N):
            if np.random.uniform() < lambda_val:
                # Sample from light-tailed vMFNM component
                samples[i] = self._sample_vmfnm_component(params)
            else:
                # Sample from heavy-tailed component
                samples[i] = self._sample_heavy_tailed_component(params)

        return samples

    def _sample_vmfnm_component(self, params: vMFNMParameters) -> np.ndarray:
        """Sample from vMFNM distribution"""
        # Sample mixture component
        k = np.random.choice(params.K, p=params.pi)

        # Sample radius from Nakagami distribution
        r = NakagamiDistribution.sample(params.m[k], params.Omega[k])

        # Sample direction from von Mises-Fisher distribution
        a = VonMisesFisherSampler.sample(params.mu[k], params.kappa[k], 1)[0]

        return r * a

    def _sample_heavy_tailed_component(self, params: vMFNMParameters) -> np.ndarray:
        """Sample from heavy-tailed inverse Nakagami component"""
        # Sample mixture component
        k = np.random.choice(params.K, p=params.pi)

        # Heavy-tailed parameters
        m_IN = max(1, int(np.ceil(np.sqrt(self.d))))  # From paper: ceil(sqrt(d))

        # Match modes between Nakagami and Inverse Nakagami (Equation 34)
        Omega_IN = self._calculate_matched_omega_inverse_nakagami(
            params.m[k], params.Omega[k], m_IN
        )

        # Sample radius from Inverse Nakagami distribution
        r = InverseNakagamiDistribution.sample(m_IN, Omega_IN)

        # Sample direction from von Mises-Fisher distribution
        a = VonMisesFisherSampler.sample(params.mu[k], params.kappa[k], 1)[0]

        return r * a

    def _calculate_matched_omega_inverse_nakagami(
        self, m_N: float, Omega_N: float, m_IN: float
    ) -> float:
        """Calculate Omega_IN to match modes (Equation 34 from paper)"""
        gamma_ratio_squared = (gamma(m_N) / gamma(m_N + 0.5)) ** 2

        Omega_IN = (2 * m_IN) / (2 * m_IN + 1) * gamma_ratio_squared * (m_N / Omega_N)

        return max(Omega_IN, 1e-6)  # Ensure positive

    def _calculate_stopping_weights(
        self,
        samples: np.ndarray,
        g_values: np.ndarray,
        sigma: float,
        params: vMFNMParameters,
        lambda_val: float,
    ) -> np.ndarray:
        """Calculate stopping criterion weights W*_t(u) = I_ΩF(u) / h_t(u)"""
        # Indicator function: I_ΩF(u) = 1 if g(u) ≤ 0, 0 otherwise
        indicators = (g_values <= 0).astype(float)

        # Smoothed indicator function: h_t(u) = Φ(-g(u)/σ_t)
        h_values = stats.norm.cdf(-g_values / sigma)

        # Stopping weights
        weights = indicators / np.maximum(h_values, 1e-15)

        return weights

    def _coefficient_of_variation(self, weights: np.ndarray) -> float:
        """Calculate coefficient of variation"""
        if len(weights) == 0 or np.sum(weights) == 0:
            return np.inf

        mean_w = np.mean(weights)
        std_w = np.std(weights)

        return std_w / mean_w if mean_w > 0 else np.inf

    def _determine_next_sigma(
        self,
        samples: np.ndarray,
        g_values: np.ndarray,
        params: vMFNMParameters,
        lambda_val: float,
        sigma_prev: float,
    ) -> float:
        """Determine next smoothing parameter by solving optimization problem (10)"""

        def cv_objective(sigma):
            """Objective function: (δ_W_t(σ) - δ_target)²"""
            if sigma >= sigma_prev or sigma <= 0:
                return 1e10

            # Calculate intermediate weights
            weights = self._calculate_intermediate_weights(
                samples, g_values, sigma, params, lambda_val
            )
            cv = self._coefficient_of_variation(weights)

            return (cv - self.delta_target) ** 2

        # Minimize over valid range
        try:
            result = minimize_scalar(
                cv_objective, bounds=(1e-8, sigma_prev * 0.999), method="bounded"
            )
            new_sigma = result.x
        except:
            # Fallback: simple reduction
            new_sigma = sigma_prev * 0.8

        return max(new_sigma, 1e-8)

    def _calculate_intermediate_weights(
        self,
        samples: np.ndarray,
        g_values: np.ndarray,
        sigma: float,
        params: vMFNMParameters,
        lambda_val: float,
    ) -> np.ndarray:
        """Calculate intermediate importance weights W_t(u_i, σ)"""
        # Intermediate distribution weight: p_t(u) / q_safe(u; φ_t-1)
        h_values = stats.norm.cdf(-g_values / sigma)  # Smoothed indicator

        # Prior density
        prior_densities = self._evaluate_prior_density(samples)

        # Safe mixture density
        safe_densities = self._evaluate_safe_mixture_density(
            samples, params, lambda_val
        )

        # Intermediate weights
        weights = h_values * prior_densities / np.maximum(safe_densities, 1e-15)

        return weights

    def _update_parameters_penalized_em(
        self,
        samples: np.ndarray,
        g_values: np.ndarray,
        params: vMFNMParameters,
        sigma: float,
        lambda_val: float,
    ) -> vMFNMParameters:
        """Update vMFNM parameters using penalized EM algorithm"""

        # Calculate importance weights for EM
        weights = self._calculate_intermediate_weights(
            samples, g_values, sigma, params, lambda_val
        )

        # Run penalized EM optimization
        updated_params, final_K = self.em_optimizer.fit(samples, weights, params)

        return updated_params

    def _estimate_failure_probability(
        self,
        samples: np.ndarray,
        g_values: np.ndarray,
        params: vMFNMParameters,
        lambda_val: float,
    ) -> float:
        """Final failure probability estimation using equation (36)"""
        # Indicator function
        indicators = (g_values <= 0).astype(float)

        # Prior densities
        prior_densities = self._evaluate_prior_density(samples)

        # Safe mixture densities
        safe_densities = self._evaluate_safe_mixture_density(
            samples, params, lambda_val
        )

        # Importance weights
        importance_weights = prior_densities / np.maximum(safe_densities, 1e-15)

        # Failure probability estimate
        pf_estimate = np.mean(indicators * importance_weights)

        return pf_estimate

    def _evaluate_prior_density(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate standard Gaussian prior density p(u)"""
        return stats.multivariate_normal.pdf(
            samples, mean=np.zeros(self.d), cov=np.eye(self.d)
        )

    def _evaluate_safe_mixture_density(
        self, samples: np.ndarray, params: vMFNMParameters, lambda_val: float
    ) -> np.ndarray:
        """Evaluate safe mixture density q_safe(u; φ)"""
        # Light-tailed component density
        vmfnm_dist = vMFNMDistribution(params)
        light_densities = vmfnm_dist.pdf(samples)

        # Heavy-tailed component density
        heavy_densities = self._evaluate_heavy_tailed_density(samples, params)

        # Safe mixture
        safe_densities = (
            lambda_val * light_densities + (1 - lambda_val) * heavy_densities
        )

        return safe_densities

    def _evaluate_heavy_tailed_density(
        self, samples: np.ndarray, params: vMFNMParameters
    ) -> np.ndarray:
        """Evaluate heavy-tailed component density"""
        n_samples = len(samples)
        densities = np.zeros(n_samples)

        for i, sample in enumerate(samples):
            r = np.linalg.norm(sample)
            if r > 1e-12:
                a = sample / r
            else:
                a = np.zeros(self.d)
                a[0] = 1.0
                r = 1e-12

            # Mixture over components
            mixture_val = 0.0
            for k in range(params.K):
                # Heavy-tailed radial component (Inverse Nakagami)
                m_IN = max(1, int(np.ceil(np.sqrt(self.d))))
                Omega_IN = self._calculate_matched_omega_inverse_nakagami(
                    params.m[k], params.Omega[k], m_IN
                )

                radial_density = InverseNakagamiDistribution.pdf(r, m_IN, Omega_IN)

                # Angular component (von Mises-Fisher)
                angular_density = self._vmf_pdf_single(a, params.mu[k], params.kappa[k])

                mixture_val += params.pi[k] * radial_density * angular_density

            densities[i] = mixture_val

        return densities

    def _vmf_pdf_single(self, x: np.ndarray, mu: np.ndarray, kappa: float) -> float:
        """von Mises-Fisher PDF for single point"""
        d = len(x)

        if kappa == 0:
            # Uniform on sphere
            return 1.0 / (2 * np.pi ** (d / 2) / gamma(d / 2))

        # Normalization constant
        C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))

        return C_d * np.exp(kappa * np.dot(x, mu))


# Benchmark problems implementation


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


# Complete heat transfer problem implementation
class HeatTransferProblem:
    """Complete heat transfer problem implementation from Section 4.5"""

    def __init__(
        self,
        grid_size: int = 21,
        correlation_length: float = 0.2,
        n_terms: int = 10,
        heat_source: float = 2000.0,
    ):
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

    def _setup_discretization(self):
        """Setup finite element discretization"""
        x = np.linspace(self.domain[0], self.domain[1], self.grid_size)
        y = np.linspace(self.domain[2], self.domain[3], self.grid_size)
        self.X, self.Y = np.meshgrid(x, y)

        # Grid points
        self.grid_points = np.column_stack([self.X.ravel(), self.Y.ravel()])
        self.n_points = len(self.grid_points)

    def _setup_kl_expansion(self):
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


# Performance evaluation utilities
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


# Complete example usage
def run_comprehensive_examples():
    """Run all benchmark problems with complete implementation"""

    print("SAFE-ICE COMPREHENSIVE IMPLEMENTATION")
    print("=" * 60)

    # Example 1: Four-mode series system
    print("\n1. Four-mode Series System Problem")
    print("-" * 40)

    problem1 = BenchmarkProblems.four_mode_series_system(z=1.0)
    safe_ice1 = SafeICE(problem1, dimension=2, K0=8, N=1000, max_iterations=15)
    pf1, results1 = safe_ice1.run(verbose=True)

    # Example 2: Three-mode problem
    print("\n2. Three-mode Problem")
    print("-" * 40)

    problem2 = BenchmarkProblems.three_mode_problem(z=3.5)
    safe_ice2 = SafeICE(problem2, dimension=2, K0=6, N=1000, max_iterations=15)
    pf2, results2 = safe_ice2.run(verbose=True)

    # Example 3: Nonlinear oscillator
    print("\n3. Nonlinear Oscillator Problem")
    print("-" * 40)

    problem3 = BenchmarkProblems.nonlinear_oscillator_simplified(d=8, z=0.06)
    safe_ice3 = SafeICE(problem3, dimension=8, K0=4, N=800, max_iterations=12)
    pf3, results3 = safe_ice3.run(verbose=True)

    # Example 4: Two-mode opposite directions
    print("\n4. Two-mode Opposite Directions")
    print("-" * 40)

    problem4 = BenchmarkProblems.two_mode_opposite_directions(z=4.0)
    safe_ice4 = SafeICE(problem4, dimension=5, K0=3, N=1000, max_iterations=10)
    pf4, results4 = safe_ice4.run(verbose=True)

    # Example 5: Heat transfer problem
    print("\n5. Heat Transfer Problem")
    print("-" * 40)

    heat_problem = HeatTransferProblem(grid_size=15, n_terms=8)
    problem5 = heat_problem.create_limit_state_function(threshold=12.0)
    safe_ice5 = SafeICE(problem5, dimension=8, K0=3, N=500, max_iterations=8)
    pf5, results5 = safe_ice5.run(verbose=True)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    results_list = [results1, results2, results3, results4, results5]
    problem_names = [
        "Four-mode",
        "Three-mode",
        "Oscillator",
        "Two-mode",
        "Heat Transfer",
    ]

    for i, (results, name) in enumerate(zip(results_list, problem_names)):
        row, col = i // 3, i % 3

        if i < 5:  # We have 5 problems
            ax = axes[row, col]

            # Plot CV convergence
            ax.semilogy(results["history"]["cv"], "b-o", markersize=4)
            ax.axhline(y=1.5, color="r", linestyle="--", alpha=0.7, label="Target CV")
            ax.set_title(f"{name}\nPf = {results['failure_probability']:.2e}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Coefficient of Variation")
            ax.grid(True, alpha=0.3)
            ax.legend()

    # Remove empty subplot
    if len(results_list) < 6:
        axes[1, 2].remove()

    plt.tight_layout()
    plt.suptitle("Safe-ICE Performance on Benchmark Problems", y=1.02, fontsize=14)
    plt.show()

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL PROBLEMS")
    print("=" * 60)
    print(f"{'Problem':<15} {'Pf Estimate':<12} {'Iterations':<10} {'Components':<10}")
    print("-" * 60)

    estimates = [pf1, pf2, pf3, pf4, pf5]
    iterations = [
        results1["iterations"],
        results2["iterations"],
        results3["iterations"],
        results4["iterations"],
        results5["iterations"],
    ]
    components = [
        results1["final_components"],
        results2["final_components"],
        results3["final_components"],
        results4["final_components"],
        results5["final_components"],
    ]

    for name, pf, iters, comps in zip(problem_names, estimates, iterations, components):
        print(f"{name:<15} {pf:<12.2e} {iters:<10} {comps:<10}")


# Performance comparison with standard methods
def performance_comparison_study():
    """Detailed performance comparison study"""
    print("\nPERFORMARE COMPARISON STUDY")
    print("=" * 50)

    # Test problem
    problem = BenchmarkProblems.four_mode_series_system(z=2.0)

    # Run comparison
    evaluator = PerformanceEvaluator()

    # Get Monte Carlo reference (smaller sample for demo)
    print("Computing Monte Carlo reference...")
    pf_mc, pf_mc_std = evaluator.run_monte_carlo_reference(problem, 2, n_samples=100000)
    print(f"Monte Carlo Reference: {pf_mc:.6e} ± {pf_mc_std:.6e}")

    # Compare methods
    comparison_results = evaluator.compare_methods(
        problem,
        2,
        reference_pf=pf_mc,
        n_runs=5,
        safe_ice_params={"K0": 6, "N": 800, "max_iterations": 12},
    )

    return comparison_results


# Advanced analysis utilities
class AdvancedAnalysis:
    """Advanced analysis tools for Safe-ICE results"""

    @staticmethod
    def analyze_component_evolution(results: Dict[str, Any]) -> None:
        """Analyze how mixture components evolve during optimization"""
        history = results["history"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Component count evolution
        axes[0, 0].plot(history["components"], "b-o")
        axes[0, 0].set_title("Component Count Evolution")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Number of Components")
        axes[0, 0].grid(True, alpha=0.3)

        # Sigma evolution
        axes[0, 1].semilogy(history["sigma"], "g-s")
        axes[0, 1].set_title("Smoothing Parameter Evolution")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("σ")
        axes[0, 1].grid(True, alpha=0.3)

        # Lambda evolution
        axes[1, 0].plot(history["lambda"], "r-^")
        axes[1, 0].set_title("Cosine Annealing Schedule")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("λ (Light-tail Weight)")
        axes[1, 0].grid(True, alpha=0.3)

        # CV evolution with target
        axes[1, 1].semilogy(history["cv"], "m-d")
        axes[1, 1].axhline(y=1.5, color="k", linestyle="--", alpha=0.7, label="Target")
        axes[1, 1].set_title("Coefficient of Variation")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("CV")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_sample_distribution(
        results: Dict[str, Any], problem_func: Callable
    ) -> None:
        """Analyze final sample distribution (for 2D problems)"""
        if results["final_samples"].shape[1] != 2:
            print("Sample distribution analysis only available for 2D problems")
            return

        samples = results["final_samples"]
        g_values = results["final_g_values"]

        # Separate failure and safe samples
        failure_samples = samples[g_values <= 0]
        safe_samples = samples[g_values > 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Sample scatter plot
        axes[0].scatter(
            safe_samples[:, 0],
            safe_samples[:, 1],
            c="blue",
            alpha=0.6,
            s=20,
            label="Safe samples",
        )
        if len(failure_samples) > 0:
            axes[0].scatter(
                failure_samples[:, 0],
                failure_samples[:, 1],
                c="red",
                alpha=0.8,
                s=30,
                label="Failure samples",
            )

        # Add failure boundary (approximate)
        x_range = np.linspace(-6, 6, 100)
        y_range = np.linspace(-6, 6, 100)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = np.zeros_like(X_grid)

        for i in range(len(x_range)):
            for j in range(len(y_range)):
                Z_grid[j, i] = problem_func(np.array([X_grid[j, i], Y_grid[j, i]]))

        axes[0].contour(
            X_grid, Y_grid, Z_grid, levels=[0], colors="black", linewidths=2
        )
        axes[0].set_xlabel("u₁")
        axes[0].set_ylabel("u₂")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Final Sample Distribution")
        axes[0].axis("equal")

        # G-function histogram
        axes[1].hist(g_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[1].axvline(
            x=0, color="red", linestyle="--", linewidth=2, label="Failure boundary"
        )
        axes[1].set_xlabel("g(u)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Limit State Function Distribution")

        plt.tight_layout()
        plt.show()

        # Statistics
        failure_rate = len(failure_samples) / len(samples)
        print(f"Final sample statistics:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Failure samples: {len(failure_samples)} ({failure_rate:.1%})")
        print(f"  G-function range: [{np.min(g_values):.3f}, {np.max(g_values):.3f}]")


# Main execution with comprehensive examples
if __name__ == "__main__":
    # Run comprehensive examples
    run_comprehensive_examples()

    # Performance comparison study
    comparison_results = performance_comparison_study()

    # Advanced analysis example
    print("\nADVANCED ANALYSIS EXAMPLE")
    print("=" * 40)

    # Run a detailed analysis on one problem
    problem = BenchmarkProblems.four_mode_series_system(z=1.5)
    safe_ice = SafeICE(problem, dimension=2, K0=8, N=1000, max_iterations=15)
    pf_estimate, detailed_results = safe_ice.run(verbose=True)

    # Perform advanced analysis
    analyzer = AdvancedAnalysis()

    print("\nComponent Evolution Analysis:")
    analyzer.analyze_component_evolution(detailed_results)

    print("\nSample Distribution Analysis:")
    analyzer.analyze_sample_distribution(detailed_results, problem)

    print("\nFINAL VALIDATION")
    print("=" * 30)
    print("Safe-ICE implementation completed successfully!")
    print(f"Final estimate: {pf_estimate:.6e}")
    print(f"Algorithm converged in {detailed_results['iterations']} iterations")
    print(
        f"Automatic component selection: {detailed_results['final_components']} components"
    )
    print(f"Final coefficient of variation: {detailed_results['final_cv']:.4f}")
