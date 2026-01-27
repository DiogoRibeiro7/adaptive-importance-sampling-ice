"""Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations."""

from .core import SafeICE, OptimizedSafeICE, AdaptiveSafeICE, vMFNMParameters
from .problems import BenchmarkProblems, HeatTransferProblem
from .analysis import PerformanceEvaluator, AdvancedAnalysis
from .distributions import (
    VonMisesFisherSampler,
    NakagamiDistribution,
    InverseNakagamiDistribution,
    vMFNMDistribution,
)
from .optimization import PenalizedEMOptimizer

__version__ = "0.1.0"
__author__ = "Diogo Ribeiro"
__all__ = [
    "SafeICE",
    "OptimizedSafeICE",
    "AdaptiveSafeICE",
    "vMFNMParameters",
    "BenchmarkProblems",
    "HeatTransferProblem",
    "PerformanceEvaluator",
    "AdvancedAnalysis",
    "VonMisesFisherSampler",
    "NakagamiDistribution",
    "InverseNakagamiDistribution",
    "vMFNMDistribution",
    "PenalizedEMOptimizer",
]
