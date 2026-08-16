# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

"""
Conditioned orthogonal layer for parameter-dependent BiLip networks.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Maintained by: Dechuan Liu (Aug 2026)
"""

from flax import linen as nn
from flax.typing import Array

from robustnn.orthogonal_jax import (
    DirectOrthogonalParams,
    ExplicitOrthogonalParams,
    Unitary,
)


# The parameterization is identical to ``Unitary``; only its bias is supplied
# at evaluation time by a conditioning network.
DirectPOrthogonalParams = DirectOrthogonalParams
ExplicitPOrthogonalParams = ExplicitOrthogonalParams


class PUnitary(Unitary):
    """Orthogonal transform with a conditioning-dependent additive bias.

    ``b`` must have ``input_size`` features and may be broadcast across the
    leading dimensions of ``x``.
    """

    use_bias: bool = False

    @nn.compact
    def __call__(self, x: Array, b: Array) -> Array:
        return self._explicit_call(x, b, self._direct_to_explicit())

    def _explicit_call(
        self, x: Array, b: Array, explicit: ExplicitPOrthogonalParams
    ) -> Array:
        return x @ explicit.R.T + b

    def _explicit_inverse_call(
        self, y: Array, b: Array, explicit: ExplicitPOrthogonalParams
    ) -> Array:
        return (y - b) @ explicit.R

    def explicit_call(
        self, params: dict, x: Array, b: Array, explicit: ExplicitPOrthogonalParams
    ) -> Array:
        """Evaluate the conditioned transform using explicit parameters."""
        return self.apply(params, x, b, explicit, method="_explicit_call")

    def inverse_call(
        self, params: dict, y: Array, b: Array, explicit: ExplicitPOrthogonalParams
    ) -> Array:
        """Evaluate the inverse conditioned transform using explicit parameters."""
        return self.apply(params, y, b, explicit, method="_explicit_inverse_call")
