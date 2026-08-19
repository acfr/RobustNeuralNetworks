# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 
"""
Parameter-conditioned Polyak-Lojasiewicz neural network for PyTorch.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Maintained by: Dechuan Liu (Aug 2026)
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from robustnn.orthogonal_torch import Params
from robustnn.pbilipnet_torch import PBiLipNet


class PPLNet(nn.Module):
    """Quadratic potential induced by a parameter-conditioned ``PBiLipNet``."""

    def __init__(
        self,
        PBiLipBlock: PBiLipNet,
        add_constant: bool = False,
        optimal_point: torch.Tensor = None,
        c: float = 0.0,
    ):
        super().__init__()
        self.pbln = PBiLipBlock
        self.optimal_point = optimal_point
        if add_constant:
            self.c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer("c", torch.tensor(float(c)))

    def forward(
        self, x: torch.Tensor, p: torch.Tensor, x_optimal: torch.Tensor = None
    ) -> torch.Tensor:
        """Evaluate the potential, optionally around an updated optimum."""
        optimal_point = self.optimal_point if x_optimal is None else x_optimal
        f = self.pbln(x, p)
        if optimal_point is not None:
            f = f - self.pbln(optimal_point, p)
        return 0.5 * torch.square(f).sum(dim=-1) + self.c

    def direct_to_explicit(self, x_optimal=None) -> Params:
        """Convert the conditioned network core to explicit NumPy parameters."""
        optimal_point = self.optimal_point if x_optimal is None else x_optimal
        if isinstance(optimal_point, torch.Tensor):
            optimal_point = optimal_point.numpy(force=True)
        lipmin, lipmax, distortion = self.pbln.get_bounds()
        return Params(
            bilip_layer=self.pbln.direct_to_explicit(),
            c=self.c.numpy(force=True),
            optimal_point=optimal_point,
            lipmin=lipmin,
            lipmax=lipmax,
            distortion=distortion,
        )

    def explicit_call(
        self,
        x: np.ndarray,
        p: np.ndarray,
        explicit: Params,
        act_mon: Callable = lambda value: np.maximum(0, value),
    ) -> np.ndarray:
        """Evaluate the potential using explicit NumPy core parameters."""
        f = self.pbln.explicit_call(x, p, explicit.bilip_layer, act_mon)
        if explicit.optimal_point is not None:
            f = f - self.pbln.explicit_call(
                explicit.optimal_point, p, explicit.bilip_layer, act_mon
            )
        return 0.5 * np.square(f).sum(axis=-1) + explicit.c

    def get_bounds(self):
        """Return the lower bound, upper bound, and distortion."""
        return self.pbln.get_bounds()
