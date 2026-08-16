# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 
"""
Conditioned orthogonal layer for parameter-dependent BiLip networks.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Maintained by: Dechuan Liu (Aug 2026)
"""

import numpy as np
import torch
import torch.nn.functional as F

from robustnn.orthogonal_torch import Params, Unitary, cayley, norm


class PUnitary(Unitary):
    """Cayley-parameterized orthogonal map with an externally supplied bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.Q_cached = None
            Q = cayley(self.alpha * self.weight / norm(self.weight))
        else:
            if self.Q_cached is None:
                with torch.no_grad():
                    self.Q_cached = cayley(self.alpha * self.weight / norm(self.weight))
            Q = self.Q_cached
        return F.linear(x, Q) + b

    def direct_to_explicit(self) -> Params:
        """Return the explicit orthogonal matrix without a fixed bias."""
        Q = cayley(self.alpha * self.weight / norm(self.weight, eps=0))
        return Params(Q=Q.numpy(force=True))

    def explicit_call(self, x: np.ndarray, b: np.ndarray, explicit: Params) -> np.ndarray:
        """Evaluate the conditioned transform with explicit NumPy parameters."""
        return x @ explicit.Q.T + b

    def inverse(self, y: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Invert the conditioned transform for a fixed bias."""
        return (y - b) @ self.direct_to_explicit().Q
