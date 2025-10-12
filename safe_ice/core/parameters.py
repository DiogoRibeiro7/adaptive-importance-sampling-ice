"""Parameter dataclasses for Safe-ICE algorithm."""

from dataclasses import dataclass
import numpy as np


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
