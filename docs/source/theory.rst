Mathematical Theory
===================

This section provides the mathematical foundation of the Safe Cross-Entropy (Safe-CE) method implemented in Safe-ICE.

Problem Formulation
-------------------

Rare Event Simulation
~~~~~~~~~~~~~~~~~~~~~

We consider the problem of estimating the failure probability:

.. math::

   p_f = P(g(\mathbf{U}) \leq 0) = \int_{\{g(\mathbf{u}) \leq 0\}} \phi(\mathbf{u}) d\mathbf{u}

where:

* :math:`\mathbf{U} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})` is a standard normal random vector
* :math:`g: \mathbb{R}^d \rightarrow \mathbb{R}` is the limit state function
* :math:`\phi(\cdot)` is the standard normal PDF
* The failure domain is :math:`\mathcal{F} = \{\mathbf{u} : g(\mathbf{u}) \leq 0\}`

For rare events, :math:`p_f` is typically very small (e.g., :math:`10^{-6}` to :math:`10^{-9}`), making direct Monte Carlo simulation inefficient.

Importance Sampling
~~~~~~~~~~~~~~~~~~~

Importance sampling estimates :math:`p_f` using an auxiliary density :math:`h(\mathbf{u})`:

.. math::

   p_f = \int_{\{g(\mathbf{u}) \leq 0\}} \frac{\phi(\mathbf{u})}{h(\mathbf{u})} h(\mathbf{u}) d\mathbf{u} = \mathbb{E}_h\left[\mathbb{I}_{g(\mathbf{U}) \leq 0} \frac{\phi(\mathbf{U})}{h(\mathbf{U})}\right]

The estimator is:

.. math::

   \hat{p}_f = \frac{1}{N} \sum_{i=1}^N \mathbb{I}_{g(\mathbf{u}_i) \leq 0} w(\mathbf{u}_i)

where :math:`\mathbf{u}_i \sim h` and :math:`w(\mathbf{u}_i) = \phi(\mathbf{u}_i)/h(\mathbf{u}_i)` are importance weights.

von Mises-Fisher Nakagami Mixture (vMFNM)
------------------------------------------

Parameterization
~~~~~~~~~~~~~~~~

Safe-ICE uses a vMFNM distribution as the importance density:

.. math::

   h(\mathbf{u}) = \sum_{k=1}^K \pi_k \cdot h_k(\mathbf{u})

Each component :math:`h_k` combines:

1. **Angular part**: von Mises-Fisher distribution on the unit sphere
2. **Radial part**: Nakagami or Inverse-Nakagami distribution

Component Density
~~~~~~~~~~~~~~~~~

For the standard vMFNM:

.. math::

   h_k(\mathbf{u}) = \text{vMF}(\mathbf{u}/\|\mathbf{u}\|; \boldsymbol{\mu}_k, \kappa_k) \cdot \text{Nakagami}(\|\mathbf{u}\|; m_k, \Omega_k)

For heavy-tailed exploration:

.. math::

   h_k^{\text{heavy}}(\mathbf{u}) = \text{vMF}(\mathbf{u}/\|\mathbf{u}\|; \boldsymbol{\mu}_k, \kappa_k) \cdot \text{InvNakagami}(\|\mathbf{u}\|; m_k, \Omega_{\text{IN},k})

von Mises-Fisher Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The vMF distribution on the unit sphere :math:`\mathbb{S}^{d-1}` has PDF:

.. math::

   \text{vMF}(\mathbf{x}; \boldsymbol{\mu}, \kappa) = C_d(\kappa) \exp(\kappa \boldsymbol{\mu}^T \mathbf{x})

where:

* :math:`\boldsymbol{\mu} \in \mathbb{S}^{d-1}` is the mean direction
* :math:`\kappa \geq 0` is the concentration parameter
* :math:`C_d(\kappa)` is the normalization constant

Nakagami Distribution
~~~~~~~~~~~~~~~~~~~~~

The Nakagami distribution for the radius has PDF:

.. math::

   \text{Nakagami}(r; m, \Omega) = \frac{2m^m}{\Gamma(m)\Omega^m} r^{2m-1} \exp\left(-\frac{m r^2}{\Omega}\right)

Parameters:

* :math:`m \geq 0.5`: Shape parameter
* :math:`\Omega > 0`: Scale parameter

Safe Cross-Entropy Algorithm
-----------------------------

Overview
~~~~~~~~

The Safe-CE algorithm iteratively updates the importance density through:

1. **Sampling**: Generate samples from current importance density
2. **Evaluation**: Compute limit state values
3. **Selection**: Identify samples closer to failure
4. **Optimization**: Update distribution parameters
5. **Adaptation**: Adjust for heavy-tailed behavior

Rarity Parameters
~~~~~~~~~~~~~~~~~

Safe-ICE uses adaptive rarity parameters :math:`\delta_j`:

.. math::

   \delta_j = \min(\delta_{\text{target}}, \delta_{j-1} + \Delta\delta)

where:

* :math:`\delta_{\text{target}}`: Target rarity (default 4.0)
* :math:`\delta_*`: Increment threshold (default 1.5)
* :math:`\Delta\delta = \delta_* \cdot \mathbb{I}_{\{\delta_{j-1} \geq \delta_*\}}`

Penalized EM Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters are updated using penalized maximum likelihood:

.. math::

   \boldsymbol{\theta}_{j+1} = \arg\max_{\boldsymbol{\theta}} \left[ \mathcal{L}(\boldsymbol{\theta}) - \beta \mathcal{P}(\boldsymbol{\theta}) \right]

Log-likelihood:

.. math::

   \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^N w_i \log h(\mathbf{u}_i; \boldsymbol{\theta})

Penalty term for automatic component selection:

.. math::

   \mathcal{P}(\boldsymbol{\theta}) = -\sum_{k=1}^K \pi_k \log \pi_k

E-Step
~~~~~~

Compute responsibilities:

.. math::

   \gamma_{ik} = \frac{\pi_k h_k(\mathbf{u}_i)}{\sum_{j=1}^K \pi_j h_j(\mathbf{u}_i)}

M-Step
~~~~~~

Update parameters:

.. math::

   \pi_k = \frac{\sum_i w_i \gamma_{ik} - \beta}{\sum_i w_i - K\beta}

.. math::

   \boldsymbol{\mu}_k = \frac{\sum_i w_i \gamma_{ik} \mathbf{x}_i}{\|\sum_i w_i \gamma_{ik} \mathbf{x}_i\|}

where :math:`\mathbf{x}_i = \mathbf{u}_i/\|\mathbf{u}_i\|`.

Heavy-Tailed Adaptation
------------------------

Inverse-Nakagami Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For enhanced exploration, Safe-ICE switches to inverse-Nakagami:

.. math::

   \text{InvNakagami}(r; m, \Omega_{\text{IN}}) = \frac{2}{\Gamma(m)} \left(\frac{m}{\Omega_{\text{IN}}}\right)^m r^{-2m-1} \exp\left(-\frac{m}{\Omega_{\text{IN}} r^2}\right)

Parameter Mapping
~~~~~~~~~~~~~~~~~

The scale parameter is adjusted:

.. math::

   \Omega_{\text{IN},k} = \frac{m_k}{d \cdot \text{PDF}_{\chi^2_d}(\Omega_k \sigma^2)}

This ensures consistent behavior at the mode while providing heavier tails.

Convergence Criteria
--------------------

Coefficient of Variation
~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm monitors the coefficient of variation:

.. math::

   \text{CoV} = \frac{\sqrt{\text{Var}[\hat{p}_f]}}{\mathbb{E}[\hat{p}_f]}

Convergence is achieved when :math:`\text{CoV} < \text{tolerance}` or maximum iterations are reached.

Adaptive Sigma
~~~~~~~~~~~~~~

The radial scaling parameter :math:`\sigma` is updated:

.. math::

   \sigma_{j+1} = \begin{cases}
   \sigma_j \cdot 1.1 & \text{if } K_j < K_{j-1} \\
   \sigma_j / 1.1 & \text{if } K_j > K_{j-1} \\
   \sigma_j & \text{otherwise}
   \end{cases}

Final Estimator
---------------

The failure probability estimate combines all iterations:

.. math::

   \hat{p}_f = \frac{1}{N_{\text{total}}} \sum_{j=1}^{J} \sum_{i=1}^{N_j} \mathbb{I}_{g(\mathbf{u}_{ji}) \leq 0} w_{ji}

where:

* :math:`J`: Number of iterations
* :math:`N_j`: Samples in iteration :math:`j`
* :math:`w_{ji}`: Importance weight for sample :math:`i` in iteration :math:`j`

References
----------

1. Papaioannou et al. (2019). "Improved cross entropy-based importance sampling with a flexible mixture model." *Reliability Engineering & System Safety*.

2. Wang et al. (2021). "Safe Cross-Entropy Method for Rare Event Simulation." *Structural Safety*.

3. Kurtz & Song (2013). "Cross-entropy-based adaptive importance sampling using Gaussian mixture." *Structural Safety*.