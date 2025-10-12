"""von Mises-Fisher distribution sampler."""

import numpy as np


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
