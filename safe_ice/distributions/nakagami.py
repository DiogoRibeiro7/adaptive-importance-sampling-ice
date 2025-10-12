import numpy as np
import scipy.stats as stats
from scipy.special import gammaln



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
