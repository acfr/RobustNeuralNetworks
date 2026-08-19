# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 
"""
Parameter-conditioned bi-Lipschitz neural networks.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Maintained by: Dechuan Liu (Aug 2026)
"""

from typing import Callable, Optional, Sequence

import jax.numpy as jnp

from flax import linen as nn
from flax.struct import dataclass
from flax.typing import Array

from robustnn.pmonlipnet_jax import (
    DirectPMonLipParams,
    ExplicitInversePMonLipParams,
    ExplicitPMonLipParams,
    PMonLipNet,
    _bounds,
)
from robustnn.porthogonal_jax import (
    DirectPOrthogonalParams,
    ExplicitPOrthogonalParams,
    PUnitary,
)


@dataclass
class DirectPBiLipParams:
    """Direct parameters of the conditioned orthogonal and monotone layers."""

    monlip_layers: Sequence[DirectPMonLipParams]
    unitary_layers: Sequence[DirectPOrthogonalParams]


@dataclass
class ExplicitPBiLipParams:
    """Explicit core parameters of a parameter-conditioned BiLipNet."""

    monlip_layers: Sequence[ExplicitPMonLipParams]
    unitary_layers: Sequence[ExplicitPOrthogonalParams]
    lipmin: float
    lipmax: float
    distortion: float


@dataclass
class ExplicitInversePBiLipParams:
    """Explicit parameters for the inverse conditioned network."""

    monlip_layers: Sequence[ExplicitInversePMonLipParams]
    unitary_layers: Sequence[ExplicitPOrthogonalParams]
    lipmin: float
    lipmax: float
    distortion: float


class _Conditioner(nn.Module):
    """Small MLP that maps the conditioning input to an affine offset."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, p: Array) -> Array:
        for features in self.features[:-1]:
            p = nn.relu(nn.Dense(features)(p))
        return nn.Dense(self.features[-1])(p)


class PBiLipNet(nn.Module):
    """BiLipNet whose affine offsets are functions of a conditioning input.

    ``po_units`` and ``pb_units`` specify the hidden and output widths of the
    orthogonal- and monotone-bias MLPs respectively.  Their final widths must
    be ``input_size`` and ``sum(units)``.
    """

    input_size: int
    units: Sequence[int]
    po_units: Sequence[int]
    pb_units: Sequence[int]
    tau: float = 10.0
    mu: float = 0.1
    nu: float = 10.0
    is_mu_fixed: bool = False
    is_nu_fixed: bool = False
    is_tau_fixed: bool = False
    act_fn: Callable = nn.relu
    depth: int = 2

    def setup(self):
        if self.depth < 1:
            raise ValueError("depth must be at least one.")
        if not self.po_units or self.po_units[-1] != self.input_size:
            raise ValueError("po_units must end with input_size.")
        if not self.pb_units or self.pb_units[-1] != sum(self.units):
            raise ValueError("pb_units must end with sum(units).")

        mu, nu, tau = _bounds(self)
        layer_tau = tau ** (1 / self.depth)
        layer_mu = mu ** (1 / self.depth)
        layer_nu = nu ** (1 / self.depth)

        self.uni = [
            PUnitary(input_size=self.input_size, name=f"uni_{index}")
            for index in range(self.depth + 1)
        ]
        self.mon = [
            PMonLipNet(
                input_size=self.input_size,
                units=self.units,
                tau=layer_tau,
                mu=layer_mu,
                nu=layer_nu,
                is_mu_fixed=self.is_mu_fixed,
                is_nu_fixed=self.is_nu_fixed,
                is_tau_fixed=self.is_tau_fixed,
                act_fn=self.act_fn,
                name=f"mon_{index}",
            )
            for index in range(self.depth)
        ]
        self.uni_b = [
            _Conditioner(self.po_units, name=f"uni_b_{index}")
            for index in range(self.depth + 1)
        ]
        self.mon_b = [
            _Conditioner(self.pb_units, name=f"mon_b_{index}")
            for index in range(self.depth)
        ]
        self.direct = DirectPBiLipParams(
            monlip_layers=[layer.direct for layer in self.mon],
            unitary_layers=[layer.direct for layer in self.uni],
        )

    def _direct_to_explicit(self) -> ExplicitPBiLipParams:
        lipmin, lipmax, distortion = self._get_bounds()
        return ExplicitPBiLipParams(
            monlip_layers=[layer._direct_to_explicit() for layer in self.mon],
            unitary_layers=[layer._direct_to_explicit() for layer in self.uni],
            lipmin=lipmin,
            lipmax=lipmax,
            distortion=distortion,
        )

    def _direct_to_explicit_inverse(
        self,
        alphas: Optional[Sequence[float]] = None,
        inverse_activation_fns: Optional[Sequence[Callable]] = None,
        iterations: Optional[Sequence[int]] = None,
        Lambdas: Optional[Sequence[float]] = None,
        tolerances: Optional[Sequence[float]] = None,
    ) -> ExplicitInversePBiLipParams:
        alphas = [None] * self.depth if alphas is None else alphas
        inverse_activation_fns = (
            [nn.relu] * self.depth
            if inverse_activation_fns is None else inverse_activation_fns
        )
        iterations = [2000] * self.depth if iterations is None else iterations
        Lambdas = [1.0] * self.depth if Lambdas is None else Lambdas
        tolerances = [1e-6] * self.depth if tolerances is None else tolerances
        arguments = {
            "alphas": alphas,
            "inverse_activation_fns": inverse_activation_fns,
            "iterations": iterations,
            "Lambdas": Lambdas,
            "tolerances": tolerances,
        }
        for name, values in arguments.items():
            if len(values) != self.depth:
                raise ValueError(f"{name} must contain {self.depth} values.")

        lipmin, lipmax, distortion = self._get_bounds()
        return ExplicitInversePBiLipParams(
            monlip_layers=[
                layer._direct_to_explicit_inverse(
                    alphas[index],
                    inverse_activation_fns[index],
                    iterations[index],
                    Lambdas[index],
                    tolerances[index],
                )
                for index, layer in enumerate(self.mon)
            ],
            unitary_layers=[layer._direct_to_explicit() for layer in self.uni],
            lipmin=lipmin,
            lipmax=lipmax,
            distortion=distortion,
        )

    def _explicit_call(self, x: Array, p: Array, explicit: ExplicitPBiLipParams) -> Array:
        for index in range(self.depth):
            x = self.uni[index]._explicit_call(
                x, self.uni_b[index](p), explicit.unitary_layers[index]
            )
            x = self.mon[index]._explicit_call(
                x, self.mon_b[index](p), explicit.monlip_layers[index]
            )
        return self.uni[self.depth]._explicit_call(
            x, self.uni_b[self.depth](p), explicit.unitary_layers[self.depth]
        )

    @nn.compact
    def __call__(self, x: Array, p: Array) -> Array:
        return self._explicit_call(x, p, self._direct_to_explicit())

    def _explicit_inverse_call(
        self, y: Array, p: Array, explicit: ExplicitInversePBiLipParams
    ) -> Array:
        return self._explicit_inverse_call_with_diagnostics(y, p, explicit)[0]

    def _explicit_inverse_call_with_diagnostics(
        self, y: Array, p: Array, explicit: ExplicitInversePBiLipParams
    ):
        residuals = [None] * self.depth
        iterations = [None] * self.depth
        for index in range(self.depth, 0, -1):
            y = self.uni[index]._explicit_inverse_call(
                y, self.uni_b[index](p), explicit.unitary_layers[index]
            )
            y, residuals[index - 1], iterations[index - 1] = (
                self.mon[index - 1]._explicit_inverse_call_with_diagnostics(
                    y,
                    self.mon_b[index - 1](p),
                    explicit.monlip_layers[index - 1],
                )
            )
        y = self.uni[0]._explicit_inverse_call(
            y, self.uni_b[0](p), explicit.unitary_layers[0]
        )
        return y, jnp.stack(residuals), jnp.stack(iterations)

    def _get_bounds(self):
        lipmin, lipmax, distortion = 1.0, 1.0, 1.0
        for layer in self.mon:
            mu, nu, tau = layer._get_bounds()
            lipmin *= mu
            lipmax *= nu
            distortion *= tau
        return lipmin, lipmax, distortion

    def get_bounds(self, params: dict = None) -> tuple:
        """Return the lower bound, upper bound, and distortion."""
        return self.apply(params, method="_get_bounds")

    def explicit_call(
        self, params: dict, x: Array, p: Array, explicit: ExplicitPBiLipParams
    ) -> Array:
        """Evaluate the conditioned network using explicit core parameters."""
        return self.apply(params, x, p, explicit, method="_explicit_call")

    def direct_to_explicit(self, params: dict) -> ExplicitPBiLipParams:
        """Convert the orthogonal and monotone core to explicit parameters."""
        return self.apply(params, method="_direct_to_explicit")

    def inverse_call(
        self, params: dict, y: Array, p: Array, explicit: ExplicitInversePBiLipParams
    ) -> Array:
        """Evaluate the inverse conditioned network using explicit parameters."""
        return self.apply(params, y, p, explicit, method="_explicit_inverse_call")

    def inverse_call_with_diagnostics(
        self, params: dict, y: Array, p: Array, explicit: ExplicitInversePBiLipParams
    ):
        """Evaluate the inverse and return each block's DYS diagnostics."""
        return self.apply(
            params, y, p, explicit, method="_explicit_inverse_call_with_diagnostics"
        )

    def direct_to_explicit_inverse(
        self,
        params: dict,
        alphas: Optional[Sequence[float]] = None,
        inverse_activation_fns: Optional[Sequence[Callable]] = None,
        iterations: Optional[Sequence[int]] = None,
        Lambdas: Optional[Sequence[float]] = None,
        tolerances: Optional[Sequence[float]] = None,
    ) -> ExplicitInversePBiLipParams:
        """Convert the conditioned network for adaptive inverse evaluation."""
        return self.apply(
            params,
            alphas,
            inverse_activation_fns,
            iterations,
            Lambdas,
            tolerances,
            method="_direct_to_explicit_inverse",
        )
