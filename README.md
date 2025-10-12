# Adaptive Importance Sampling ICE

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2509.07160-b31b1b.svg)](https://arxiv.org/abs/2509.07160)

Complete Python implementation of **Safe Cross-Entropy-Based Importance Sampling** for rare event simulations in reliability analysis.

## 📖 Overview

This repository provides a rigorous implementation of the Safe-ICE algorithm from:

> **Gao, Z., & Karniadakis, G. (2025).** Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations. *arXiv preprint arXiv:2509.07160*.

Safe-ICE addresses the challenge of estimating extremely small failure probabilities (down to 10⁻¹⁰ and beyond) in high-dimensional reliability problems through three key innovations:

1. **Penalized EM Algorithm**: Automatically determines the optimal number of mixture components
2. **Heavy-Tailed Exploration**: Uses inverse Nakagami distributions to explore rare failure regions
3. **Adaptive Annealing**: Smoothly transitions from exploration to exploitation via cosine annealing

## ✨ Key Features

- ✅ **Complete vMFNM Implementation**: von Mises-Fisher-Nakagami mixture with exact sampling
- ✅ **Automatic Component Selection**: Penalized EM removes redundant components during optimization
- ✅ **Heavy-Tailed Sampling**: Inverse Nakagami distribution for extreme tail exploration
- ✅ **Robust Numerical Methods**: Stable Bessel function evaluations and log-space computations
- ✅ **Comprehensive Benchmarks**: All test problems from the paper
- ✅ **PDE Solver Integration**: Heat transfer problem with Karhunen-Loève expansion
- ✅ **Advanced Analysis Tools**: Convergence visualization and performance metrics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/diogoribeiro7/adaptive-importance-sampling-ice.git
cd adaptive-importance-sampling-ice

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Basic Usage

```python
from safe_ice import SafeICE, BenchmarkProblems

# Define a limit state function (failure when g(u) ≤ 0)
problem = BenchmarkProblems.four_mode_series_system(z=2.0)

# Initialize Safe-ICE
safe_ice = SafeICE(
    limit_state_function=problem,
    dimension=2,
    K0=8,              # Initial mixture components
    N=1000,            # Samples per iteration
    delta_star=1.5     # CV stopping criterion
)

# Run the algorithm
pf_estimate, results = safe_ice.run(verbose=True)

print(f"Failure Probability: {pf_estimate:.6e}")
print(f"Converged in {results['iterations']} iterations")
print(f"Final components: {results['final_components']}")
```

### Example Output

```
Safe-ICE Algorithm
Problem dimension: 2
Initial components: 8
Samples per iteration: 1000
--------------------------------------------------
Iteration  1: σ=0.850000, λ=0.146, K=8
           CV=4.2341
Iteration  2: σ=0.642000, λ=0.345, K=6
           CV=3.8124
...
Iteration  5: σ=0.128000, λ=0.891, K=3
           CV=1.4523
Converged: CV 1.4523 ≤ 1.5
--------------------------------------------------
Final Results:
Failure Probability: 2.35e-04
Total Iterations: 5
Final Components: 3
Final CV: 1.4523
```

## 🔬 Advanced Features

### Performance Comparison

```python
from safe_ice import PerformanceEvaluator

evaluator = PerformanceEvaluator()

# Compare with Monte Carlo
comparison = evaluator.compare_methods(
    limit_state_func=problem,
    dimension=2,
    n_runs=10,
    safe_ice_params={'K0': 6, 'N': 1000}
)
```

### Convergence Analysis

```python
from safe_ice import AdvancedAnalysis

analyzer = AdvancedAnalysis()

# Analyze component evolution
analyzer.analyze_component_evolution(results)

# Visualize sample distribution (2D only)
analyzer.analyze_sample_distribution(results, problem)
```

### Custom Limit State Functions

```python
def my_limit_state(u):
    """
    Custom failure criterion
    Returns: g(u) where failure occurs when g(u) ≤ 0
    """
    # Your custom implementation
    return some_complex_function(u)

safe_ice = SafeICE(my_limit_state, dimension=10)
pf_estimate, results = safe_ice.run()
```


## 🎯 Algorithm Overview

### The Safe-ICE Method

Safe-ICE estimates failure probabilities P_F = P(g(U) ≤ 0) where:

- U ~ N(0, I) is a random input vector
- g(U) is the limit state function
- Failure occurs when g(U) ≤ 0

**Key Algorithm Steps:**

1. **Initialize** mixture parameters with K₀ components
2. **Generate samples** from safe mixture: q_safe = λ·q_vMFNM + (1-λ)·q_heavy
3. **Evaluate** limit state function g(u) for all samples
4. **Check convergence** using coefficient of variation of importance weights
5. **Adapt smoothing** parameter σ to control intermediate distributions
6. **Update parameters** using penalized EM to fit vMFNM mixture
7. **Repeat** until convergence

### Penalized EM Update (Equation 21)

The mixture weights are updated with a cross-entropy penalty term:

```
π_k^(new) = π_k^(EM) + β · (sum W_i) / (sum sum γ_s W_i) · π_k^(old) · [ln π_k^(old) - Σ π_s ln π_s]
```

This automatically removes redundant components during optimization.

### Heavy-Tailed Component (Equation 34)

The inverse Nakagami parameters are matched to ensure efficient exploration:

```
Ω^(IN) = (2m^(IN))/(2m^(IN) + 1) · [Γ(m^N) / Γ(m^N + 0.5)]² · (m^N / Ω^N)
```



## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=safe_ice --cov-report=html

# Run specific test
pytest tests/test_safe_ice.py::test_four_mode_convergence -v
```


## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{gao2025safe,
  title={Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations},
  author={Gao, Zhiwei and Karniadakis, George},
  journal={arXiv preprint arXiv:2509.07160},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper by Zhiwei Gao and George Karniadakis
- Built upon the ICE method by Papaioannou et al. (2019)
- Inspired by the cross-entropy method of Rubinstein & Kroese (2004)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/diogoribeiro7/adaptive-importance-sampling-ice/issues)
- **Discussions**: [GitHub Discussions](https://github.com/diogoribeiro7/adaptive-importance-sampling-ice/discussions)
- **Author**: Diogo Ribeiro
- **GitHub**: [@diogoribeiro7](https://github.com/diogoribeiro7)

## 🔗 Related Work

- [Original ICE Paper](https://arxiv.org/abs/2509.07160)
- [Cross-Entropy Method](https://link.springer.com/article/10.1007/s10479-005-5724-z)
- [Subset Simulation](https://doi.org/10.1016/S0266-8920(01)00019-4)

---

**⭐ If you find this useful, please star the repository!**
