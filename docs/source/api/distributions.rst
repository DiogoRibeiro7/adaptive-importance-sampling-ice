Distributions Module
====================

The distributions module implements the probability distributions used in Safe-ICE.

von Mises-Fisher Distribution
-----------------------------

.. automodule:: safe_ice.distributions.von_mises_fisher
   :members:
   :undoc-members:
   :show-inheritance:

Nakagami Distribution
---------------------

.. automodule:: safe_ice.distributions.nakagami
   :members:
   :undoc-members:
   :show-inheritance:

Inverse Nakagami Distribution
-----------------------------

.. automodule:: safe_ice.distributions.inverse_nakagami
   :members:
   :undoc-members:
   :show-inheritance:

vMFNM Distribution
------------------

.. automodule:: safe_ice.distributions.vmfnm
   :members:
   :undoc-members:
   :show-inheritance:

Mixture Distribution
--------------------

.. automodule:: safe_ice.distributions.mixture
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

von Mises-Fisher Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from safe_ice.distributions.von_mises_fisher import VonMisesFisherSampler
   import numpy as np

   # Create sampler
   sampler = VonMisesFisherSampler(dimension=3)

   # Sample from vMF distribution
   mu = np.array([0, 0, 1])  # North pole
   kappa = 10.0               # High concentration

   samples = sampler.sample(mu, kappa, n_samples=1000)

   # Samples are on unit sphere
   assert np.allclose(np.linalg.norm(samples, axis=1), 1.0)

Nakagami Distribution
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from safe_ice.distributions.nakagami import NakagamiDistribution

   # Create distribution
   nakagami = NakagamiDistribution(m=2.0, omega=1.0)

   # Evaluate PDF
   x = np.linspace(0, 3, 100)
   pdf_values = nakagami.pdf(x)

   # Sample
   samples = nakagami.sample(1000)

   # Compute CDF
   cdf_values = nakagami.cdf(x)

vMFNM Mixture
~~~~~~~~~~~~~

.. code-block:: python

   from safe_ice.distributions.vmfnm import vMFNMDistribution
   from safe_ice.core.parameters import vMFNMParameters
   import numpy as np

   # Create mixture parameters
   params = vMFNMParameters(
       K=2,                           # Two components
       d=3,                           # 3D space
       pi=np.array([0.6, 0.4]),      # Mixture weights
       m=np.array([2.0, 3.0]),       # Nakagami shape
       Omega=np.array([1.0, 1.5]),   # Nakagami scale
       mu=np.array([[1, 0, 0],       # Component directions
                    [0, 1, 0]]),
       kappa=np.array([5.0, 8.0])    # Concentrations
   )

   # Create distribution
   vmfnm = vMFNMDistribution(params)

   # Sample
   samples = vmfnm.sample(1000)

   # Evaluate log-likelihood
   log_prob = vmfnm.log_pdf(samples)

Heavy-Tailed Sampling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from safe_ice.distributions.inverse_nakagami import InverseNakagamiDistribution

   # Create heavy-tailed distribution
   inv_nakagami = InverseNakagamiDistribution(m=2.0, omega_in=1.0)

   # Sample (heavy-tailed)
   samples = inv_nakagami.sample(1000)

   # PDF evaluation
   x = np.linspace(0.1, 10, 100)
   pdf_values = inv_nakagami.pdf(x)

   # Note: Heavy tails decay slower than Nakagami
   print(f"Tail behavior at x=10: {pdf_values[-1]:.2e}")