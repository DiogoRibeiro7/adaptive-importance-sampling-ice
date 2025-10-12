"""Main Safe-ICE algorithm implementation."""

import numpy as np
import scipy.stats as stats
from scipy.special import gamma
from scipy.optimize import minimize_scalar
from typing import Tuple, Callable, Dict, Any

from .parameters import vMFNMParameters
from ..distributions.mixture import vMFNMDistribution
from ..distributions.nakagami import NakagamiDistribution, InverseNakagamiDistribution
from ..distributions.vmf import VonMisesFisherSampler
from ..optimization.penalized_em import PenalizedEMOptimizer


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
    ) -> None:
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

        def cv_objective(sigma: float) -> float:
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
        from scipy.special import iv

        d = len(x)

        if kappa == 0:
            # Uniform on sphere
            return 1.0 / (2 * np.pi ** (d / 2) / gamma(d / 2))

        # Normalization constant
        C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))

        return C_d * np.exp(kappa * np.dot(x, mu))
