# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 
"""
Parameter-conditioned Polyak-Lojasiewicz neural network.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Maintained by: Dechuan Liu (Aug 2026)
"""

import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass
from flax.typing import Array

from robustnn.pbilipnet_jax import (
    DirectPBiLipParams,
    ExplicitPBiLipParams,
    PBiLipNet,
)


@dataclass
class DirectPPLParams:
    """Direct parameters of a parameter-conditioned PLNet."""

    bilip_layer: DirectPBiLipParams
    c: Array = None


@dataclass
class ExplicitPPLParams:
    """Explicit parameters of a parameter-conditioned PLNet."""

    bilip_layer: ExplicitPBiLipParams
    c: Array = None
    optimal_point: Array = None
    lipmin: float = 0.1
    lipmax: float = 10.0
    distortion: float = 100.0


class PPLNet(nn.Module):
    """Quadratic potential of a parameter-conditioned ``PBiLipNet``.

    The optional optimal point is evaluated with the same conditioning input,
    guaranteeing a minimum at that point for every fixed ``p``.
    """

    PBiLipBlock: PBiLipNet
    add_constant: bool = False
    optimal_point: Array = None
    c: float = 0.0

    def setup(self):
        c = (
            self.param("c", nn.initializers.constant(0.0), (1,), jnp.float32)
            if self.add_constant
            else self.c
        )
        self.direct = DirectPPLParams(bilip_layer=self.PBiLipBlock.direct, c=c)

    def _direct_to_explicit(self, x_optimal: Array = None) -> ExplicitPPLParams:
        optimal_point = self.optimal_point if x_optimal is None else x_optimal
        lipmin, lipmax, distortion = self.PBiLipBlock._get_bounds()
        return ExplicitPPLParams(
            bilip_layer=self.PBiLipBlock._direct_to_explicit(),
            c=self.direct.c,
            optimal_point=optimal_point,
            lipmin=lipmin,
            lipmax=lipmax,
            distortion=distortion,
        )

    def _explicit_call(self, x: Array, p: Array, explicit: ExplicitPPLParams) -> Array:
        f = self.PBiLipBlock._explicit_call(x, p, explicit.bilip_layer)
        if explicit.optimal_point is not None:
            f -= self.PBiLipBlock._explicit_call(
                explicit.optimal_point, p, explicit.bilip_layer
            )
        return 0.5 * jnp.sum(jnp.square(f), axis=-1) + explicit.c

    @nn.compact
    def __call__(self, x: Array, p: Array, x_optimal: Array = None) -> Array:
        return self._explicit_call(x, p, self._direct_to_explicit(x_optimal))

    def _get_bounds(self):
        return self.PBiLipBlock._get_bounds()

    def get_bounds(self, params: dict = None) -> tuple:
        """Return the lower bound, upper bound, and distortion."""
        return self.apply(params, method="_get_bounds")

    def explicit_call(
        self, params: dict, x: Array, p: Array, explicit: ExplicitPPLParams
    ) -> Array:
        """Evaluate the conditioned potential using explicit parameters."""
        return self.apply(params, x, p, explicit, method="_explicit_call")

    def direct_to_explicit(
        self, params: dict, x_optimal: Array = None
    ) -> ExplicitPPLParams:
        """Convert the conditioned PLNet to explicit core parameters."""
        return self.apply(params, x_optimal=x_optimal, method="_direct_to_explicit")
