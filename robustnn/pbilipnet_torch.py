# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 
"""
Parameter-conditioned bi-Lipschitz neural networks for PyTorch.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Maintained by: Dechuan Liu (Aug 2026)
"""

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from robustnn.orthogonal_torch import Params
from robustnn.pmonlipnet_torch import PMonLipNet, _resolve_bounds
from robustnn.porthogonal_torch import PUnitary


class _Conditioner(nn.Module):
    """MLP mapping a conditioning input to an affine offset."""

    def __init__(self, features: Sequence[int]):
        super().__init__()
        layers = []
        for index, width in enumerate(features):
            layers.append(nn.LazyLinear(width))
            if index < len(features) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.layers(p)


class PBiLipNet(nn.Module):
    """BiLipNet with orthogonal and hidden biases conditioned on ``p``."""

    def __init__(
        self,
        features: int,
        unit_features: Sequence[int],
        po_units: Sequence[int],
        pb_units: Sequence[int],
        mu: float = None,
        nu: float = None,
        tau: float = None,
        is_mu_fixed: bool = False,
        is_nu_fixed: bool = False,
        is_tau_fixed: bool = False,
        depth: int = 1,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one.")
        if not po_units or po_units[-1] != features:
            raise ValueError("po_units must end with features.")
        if not pb_units or pb_units[-1] != sum(unit_features):
            raise ValueError("pb_units must end with sum(unit_features).")

        mu, nu, tau = _resolve_bounds(mu, nu, tau)
        layer_mu, layer_nu = mu ** (1.0 / depth), nu ** (1.0 / depth)
        layer_tau = tau ** (1.0 / depth) if is_tau_fixed else layer_nu / layer_mu
        self.depth = depth
        self.orth_layers = nn.ModuleList(PUnitary(features, features) for _ in range(depth + 1))
        self.mon_layers = nn.ModuleList(
            PMonLipNet(
                features, unit_features, layer_mu, layer_nu, layer_tau,
                is_mu_fixed, is_nu_fixed, is_tau_fixed, act,
            )
            for _ in range(depth)
        )
        self.orth_biases = nn.ModuleList(_Conditioner(po_units) for _ in range(depth + 1))
        self.mon_biases = nn.ModuleList(_Conditioner(pb_units) for _ in range(depth))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        for index in range(self.depth):
            x = self.orth_layers[index](x, self.orth_biases[index](p))
            x = self.mon_layers[index](x, self.mon_biases[index](p))
        return self.orth_layers[self.depth](x, self.orth_biases[self.depth](p))

    def direct_to_explicit(self) -> Params:
        """Convert the orthogonal and monotone cores to explicit parameters."""
        lipmin, lipmax, distortion = self.get_bounds()
        return Params(
            monlip_layers=[layer.direct_to_explicit() for layer in self.mon_layers],
            unitary_layers=[layer.direct_to_explicit() for layer in self.orth_layers],
            lipmin=lipmin,
            lipmax=lipmax,
            distortion=distortion,
        )

    @staticmethod
    def _condition(module: nn.Module, p: np.ndarray) -> np.ndarray:
        parameter = next(module.parameters())
        with torch.no_grad():
            value = module(torch.as_tensor(p, dtype=parameter.dtype, device=parameter.device))
        return value.numpy(force=True)

    def explicit_call(
        self,
        x: np.ndarray,
        p: np.ndarray,
        explicit: Params,
        act_mon: Callable = lambda value: np.maximum(0, value),
    ) -> np.ndarray:
        """Evaluate the conditioned network with explicit NumPy core parameters."""
        for index in range(self.depth):
            x = self.orth_layers[index].explicit_call(
                x, self._condition(self.orth_biases[index], p), explicit.unitary_layers[index]
            )
            x = self.mon_layers[index].explicit_call(
                x, self._condition(self.mon_biases[index], p), explicit.monlip_layers[index], act_mon
            )
        return self.orth_layers[self.depth].explicit_call(
            x, self._condition(self.orth_biases[self.depth], p), explicit.unitary_layers[self.depth]
        )

    def get_bounds(self):
        """Return the lower bound, upper bound, and distortion."""
        lipmin, lipmax, distortion = 1.0, 1.0, 1.0
        for layer in self.mon_layers:
            mu, nu, tau = layer.get_bounds()
            lipmin, lipmax, distortion = lipmin * mu, lipmax * nu, distortion * tau
        return lipmin, lipmax, distortion

    def inverse(
        self,
        y: np.ndarray,
        p: np.ndarray,
        alphas: Optional[Sequence[float]] = None,
        inverse_activation_fns: Optional[Sequence[Callable]] = None,
        iterations: Optional[Sequence[int]] = None,
        Lambdas: Optional[Sequence[float]] = None,
        tolerances: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Invert the conditioned network for a fixed conditioning input."""
        return self.inverse_with_diagnostics(
            y, p, alphas, inverse_activation_fns, iterations, Lambdas, tolerances
        )[0]

    def inverse_with_diagnostics(
        self,
        y: np.ndarray,
        p: np.ndarray,
        alphas: Optional[Sequence[float]] = None,
        inverse_activation_fns: Optional[Sequence[Callable]] = None,
        iterations: Optional[Sequence[int]] = None,
        Lambdas: Optional[Sequence[float]] = None,
        tolerances: Optional[Sequence[float]] = None,
    ):
        """Invert the network and return per-block DYS diagnostics."""
        values = {
            "alphas": [None] * self.depth if alphas is None else alphas,
            "inverse_activation_fns": (
                [lambda value: np.maximum(0, value)] * self.depth
                if inverse_activation_fns is None else inverse_activation_fns
            ),
            "iterations": [2000] * self.depth if iterations is None else iterations,
            "Lambdas": [1.0] * self.depth if Lambdas is None else Lambdas,
            "tolerances": [1e-6] * self.depth if tolerances is None else tolerances,
        }
        for name, sequence in values.items():
            if len(sequence) != self.depth:
                raise ValueError(f"{name} must contain {self.depth} values.")

        residuals = [None] * self.depth
        steps = [None] * self.depth
        effective_alphas = [None] * self.depth
        for index in range(self.depth, 0, -1):
            y = self.orth_layers[index].inverse(y, self._condition(self.orth_biases[index], p))
            y, residuals[index - 1], steps[index - 1], effective_alphas[index - 1] = (
                self.mon_layers[index - 1].inverse_with_diagnostics(
                    y,
                    self._condition(self.mon_biases[index - 1], p),
                    alpha=values["alphas"][index - 1],
                    inverse_activation_fn=values["inverse_activation_fns"][index - 1],
                    iterations=values["iterations"][index - 1],
                    Lambda=values["Lambdas"][index - 1],
                    tolerance=values["tolerances"][index - 1],
                )
            )
        y = self.orth_layers[0].inverse(y, self._condition(self.orth_biases[0], p))
        return (
            y,
            np.asarray(residuals),
            np.asarray(steps),
            np.asarray(effective_alphas),
        )
