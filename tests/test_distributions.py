"""Tests for distribution implementations."""

import numpy as np
import pytest
from scipy.stats import kstest, chi2

from safe_ice.distributions.vmf import VonMisesFisherSampler
from safe_ice.distributions.nakagami import (
    NakagamiDistribution,
    InverseNakagamiDistribution,
)
from safe_ice.distributions.mixture import vMFNMDistribution
from safe_ice.core.parameters import vMFNMParameters



class TestVonMisesFisherSampler:
    """Tests for von Mises-Fisher distribution sampler."""

    def test_sample_shape(self):
        """Test that samples have correct shape."""
        mu = np.array([1.0, 0.0, 0.0])
        kappa = 2.0
        n_samples = 100

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples)

        assert samples.shape == (n_samples, 3)

    def test_samples_are_unit_vectors(self):
        """Test that all samples are unit vectors."""
        mu = np.array([1.0, 0.0])
        kappa = 5.0
        n_samples = 50

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples)
        norms = np.linalg.norm(samples, axis=1)

        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_zero_kappa_uniform(self):
        """Test that kappa=0 gives uniform distribution."""
        mu = np.array([1.0, 0.0])
        kappa = 0.0
        n_samples = 1000

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples)

        # Check mean direction is close to zero
        mean_direction = np.mean(samples, axis=0)
        assert np.linalg.norm(mean_direction) < 0.2

    def test_high_kappa_concentration(self):
        """Test that high kappa concentrates around mean."""
        mu = np.array([1.0, 0.0, 0.0])
        kappa = 50.0
        n_samples = 100

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples)

        # All samples should be close to mu
        dot_products = np.dot(samples, mu)
        assert np.all(dot_products > 0.9)

    def test_circular_vmf(self):
        """Test 2D circular case."""
        mu = np.array([1.0, 0.0])
        kappa = 3.0
        n_samples = 200

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples)

        assert samples.shape == (n_samples, 2)
        norms = np.linalg.norm(samples, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)


class TestNakagamiDistribution:
    """Tests for Nakagami distribution."""

    def test_pdf_positive(self):
        """Test that PDF is positive for r > 0."""
        m, Omega = 2.0, 1.0
        r = np.linspace(0.1, 5.0, 50)

        pdf_values = NakagamiDistribution.pdf(r, m, Omega)

        assert np.all(pdf_values >= 0)

    def test_pdf_zero_at_zero(self):
        """Test that PDF is zero at r = 0."""
        m, Omega = 2.0, 1.0

        pdf_value = NakagamiDistribution.pdf(0.0, m, Omega)

        assert pdf_value == 0.0

    def test_sample_shape(self):
        """Test sample shape."""
        m, Omega = 1.5, 2.0
        n_samples = 100

        samples = NakagamiDistribution.sample(m, Omega, n_samples)

        assert samples.shape == (n_samples,)
        assert np.all(samples > 0)

    def test_sample_mean_approximation(self):
        """Test that sample mean approximates theoretical mean."""
        m, Omega = 2.0, 1.0
        n_samples = 10000

        samples = NakagamiDistribution.sample(m, Omega, n_samples)
        sample_mean = np.mean(samples)

        # Theoretical mean: E[R] = (Gamma(m + 0.5) / Gamma(m)) * sqrt(Omega / m)
        from scipy.special import gamma

        theoretical_mean = (gamma(m + 0.5) / gamma(m)) * np.sqrt(Omega / m)

        # Allow 5% error due to sampling
        np.testing.assert_allclose(sample_mean, theoretical_mean, rtol=0.05)

    def test_cdf_bounds(self):
        """Test CDF is between 0 and 1."""
        m, Omega = 2.0, 1.0
        r = np.linspace(0, 5, 50)

        cdf_values = NakagamiDistribution.cdf(r, m, Omega)

        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)
        assert np.allclose(cdf_values[-1], 1.0, atol=0.01)


class TestInverseNakagamiDistribution:
    """Tests for Inverse Nakagami distribution."""

    def test_pdf_positive(self):
        """Test that PDF is positive for r > 0."""
        m, Omega = 2.0, 1.0
        r = np.linspace(0.1, 5.0, 50)

        pdf_values = InverseNakagamiDistribution.pdf(r, m, Omega)

        assert np.all(pdf_values >= 0)

    def test_sample_positive(self):
        """Test that all samples are positive."""
        m, Omega = 1.5, 2.0
        n_samples = 100

        samples = InverseNakagamiDistribution.sample(m, Omega, n_samples)

        assert samples.shape == (n_samples,)
        assert np.all(samples > 0)

    def test_heavy_tail_property(self):
        """Test that distribution has heavy tails."""
        m, Omega = 2.0, 1.0
        n_samples = 10000

        samples = InverseNakagamiDistribution.sample(m, Omega, n_samples)

        # Heavy-tailed: should have some very large values
        assert np.max(samples) > 5.0
        # Should also have small values
        assert np.min(samples) < 0.5


class TestvMFNMDistribution:
    """Tests for vMFNM mixture distribution."""

    @pytest.fixture
    def simple_params_2d(self):
        """Simple 2D vMFNM parameters."""
        K = 2
        d = 2
        return vMFNMParameters(
            pi=np.array([0.6, 0.4]),
            m=np.array([2.0, 1.5]),
            Omega=np.array([1.0, 1.5]),
            mu=np.array([[1.0, 0.0], [0.0, 1.0]]),
            kappa=np.array([3.0, 2.0]),
        )

    def test_initialization(self, simple_params_2d):
        """Test distribution initialization."""
        dist = vMFNMDistribution(simple_params_2d)

        assert dist.params.K == 2
        assert dist.params.d == 2

    def test_pdf_positive(self, simple_params_2d):
        """Test that PDF is non-negative."""
        dist = vMFNMDistribution(simple_params_2d)

        x = np.random.randn(10, 2)
        pdf_values = dist.pdf(x)

        assert np.all(pdf_values >= 0)

    def test_sample_shape(self, simple_params_2d):
        """Test sample shape."""
        dist = vMFNMDistribution(simple_params_2d)
        n_samples = 100

        samples = dist.sample(n_samples)

        assert samples.shape == (n_samples, 2)

    def test_sample_pdf_consistency(self, simple_params_2d):
        """Test that samples have reasonable PDF values."""
        dist = vMFNMDistribution(simple_params_2d)
        n_samples = 50

        samples = dist.sample(n_samples)
        pdf_values = dist.pdf(samples)

        # All PDF values should be positive
        assert np.all(pdf_values > 0)

    def test_log_likelihood(self, simple_params_2d):
        """Test log-likelihood computation."""
        dist = vMFNMDistribution(simple_params_2d)

        samples = dist.sample(100)
        log_likelihood = dist.log_likelihood(samples)

        # Should be finite and negative
        assert np.isfinite(log_likelihood)
        assert log_likelihood < 0

    @pytest.mark.parametrize("dimension", [2, 5, 10])
    def test_different_dimensions(self, dimension):
        """Test distribution works in different dimensions."""
        K = 3
        params = vMFNMParameters(
            pi=np.ones(K) / K,
            m=np.random.uniform(1, 3, K),
            Omega=np.random.uniform(0.5, 2, K),
            mu=np.random.randn(K, dimension),
            kappa=np.random.uniform(0.5, 3, K),
        )

        dist = vMFNMDistribution(params)
        samples = dist.sample(50)

        assert samples.shape == (50, dimension)


class TestIntegration:
    """Integration tests for distributions."""

    def test_vmfnm_mixture_normalization(self):
        """Test that vMFNM PDF approximately integrates to 1."""
        # This is a Monte Carlo integration test
        params = vMFNMParameters(
            pi=np.array([1.0]),
            m=np.array([2.0]),
            Omega=np.array([1.0]),
            mu=np.array([[1.0, 0.0]]),
            kappa=np.array([2.0]),
        )

        dist = vMFNMDistribution(params)

        # Generate many samples
        n_samples = 100000
        samples = np.random.randn(n_samples, 2) * 3  # Cover wide range

        # Approximate integral using importance sampling
        pdf_values = dist.pdf(samples)
        prior_pdf = (
            1 / (2 * np.pi) * np.exp(-0.5 * np.sum(samples**2, axis=1))
        )

        # This should be roughly n_samples if PDF integrates to 1
        integral_estimate = np.sum(pdf_values / prior_pdf) / n_samples

        # Allow large tolerance due to Monte Carlo approximation
        # This is more of a sanity check
        assert 0.5 < integral_estimate < 2.0

    @pytest.mark.slow
    def test_vmfnm_sampling_consistency(self):
        """Test sampling and PDF evaluation are consistent."""
        params = vMFNMParameters(
            pi=np.array([0.5, 0.5]),
            m=np.array([2.0, 1.5]),
            Omega=np.array([1.0, 1.5]),
            mu=np.array([[1.0, 0.0], [-1.0, 0.0]]),
            kappa=np.array([3.0, 2.0]),
        )

        dist = vMFNMDistribution(params)

        # Generate samples
        samples = dist.sample(5000)

        # Samples should roughly follow the distribution
        # Test: mean of samples should be near zero (symmetric mixture)
        sample_mean = np.mean(samples, axis=0)
        assert np.linalg.norm(sample_mean) < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
