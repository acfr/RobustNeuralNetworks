# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 
"""
Conditioned monotone-Lipschitz layer for parameter-dependent BiLip nets.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Maintained by: Dechuan Liu (Aug 2026)
"""

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Array

from robustnn.monlipnet_jax import (
    DirectMonLipParams,
    ExplicitInverseMonLipParams,
    ExplicitMonLipParams,
    MonLipNet,
)


# The Cayley parameterization and explicit representation are unchanged.  The
# hidden biases are inputs rather than trainable parameters.
DirectPMonLipParams = DirectMonLipParams
ExplicitPMonLipParams = ExplicitMonLipParams
ExplicitInversePMonLipParams = ExplicitInverseMonLipParams


def _bounds(module: nn.Module):
    """Create the two independent bound parameters, when requested."""
    fixed = (module.is_mu_fixed, module.is_nu_fixed, module.is_tau_fixed)
    if fixed == (True, True, True):
        raise ValueError("Cannot fix mu, nu, and tau at the same time.")

    def learn_mu():
        return jnp.exp(
            module.param(
                "logmu", nn.initializers.constant(jnp.log(module.mu)), (1,), jnp.float32
            )
        )[-1]

    def learn_nu():
        return jnp.exp(
            module.param(
                "lognu", nn.initializers.constant(jnp.log(module.nu)), (1,), jnp.float32
            )
        )[-1]

    values = {
        (True, True, False): lambda: (module.mu, module.nu, module.nu / module.mu),
        (True, False, True): lambda: (module.mu, module.tau * module.mu, module.tau),
        (False, True, True): lambda: (module.nu / module.tau, module.nu, module.tau),
        (True, False, False): lambda: (module.mu, learn_nu(), None),
        (False, True, False): lambda: (learn_mu(), module.nu, None),
        (False, False, True): lambda: (learn_mu(), None, module.tau),
        (False, False, False): lambda: (learn_mu(), learn_nu(), None),
    }
    mu, nu, tau = values[fixed]()
    if tau is None:
        tau = nu / mu
    elif nu is None:
        nu = tau * mu
    return mu, nu, tau


class PMonLipNet(MonLipNet):
    """Monotone-Lipschitz map with hidden biases supplied as ``b``.

    ``b`` has ``sum(units)`` features, one contiguous slice per hidden layer.
    The output bias remains a trainable, parameter-independent offset.
    """

    def setup(self):
        _bounds(self)
        by = self.param(
            "by", nn.initializers.zeros_init(), (self.input_size,), jnp.float32
        )
        fq_shape = (self.input_size, sum(self.units))
        Fq = self.param("Fq", nn.initializers.glorot_normal(), fq_shape, jnp.float32)
        fq = self.param(
            "fq", nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32
        )

        Fabs, fabs, bs = [], [], []
        previous_units = 0
        for index, units in enumerate(self.units):
            Fab = self.param(
                f"Fab{index}",
                nn.initializers.glorot_normal(),
                (units + previous_units, units),
                jnp.float32,
            )
            Fabs.append(Fab)
            fabs.append(
                self.param(
                    f"fab{index}",
                    nn.initializers.constant(jnp.linalg.norm(Fab)),
                    (1,),
                    jnp.float32,
                )
            )
            # Retained only to reuse MonLipNet's explicit conversion.  These
            # constants are never applied; ``b`` replaces them in the forward map.
            bs.append(jnp.zeros((units,), dtype=jnp.float32))
            previous_units = units

        self.direct = DirectPMonLipParams(
            Fq=Fq, fq=fq, Fabs=Fabs, fabs=fabs, bs=bs, by=by
        )

    def _explicit_call(self, x: Array, b: Array, explicit: ExplicitPMonLipParams) -> Array:
        if b.shape[-1] != sum(self.units):
            raise ValueError(f"b must have {sum(self.units)} features; got {b.shape[-1]}.")

        y = explicit.mu * x + explicit.by
        z = x[..., :0]
        index = 0
        for layer, units in enumerate(self.units):
            next_index = index + units
            z = self.act_fn(
                2 * (z @ explicit.Ak_1s[layer]) @ explicit.BTks[layer]
                + explicit.sqrt_2g * x @ explicit.STks[layer]
                + b[..., index:next_index]
            )
            y += explicit.sqrt_g2 * z @ explicit.STks[layer].T
            index = next_index
        return y

    @nn.compact
    def __call__(self, x: Array, b: Array) -> Array:
        return self._explicit_call(x, b, self._direct_to_explicit())

    def _explicit_inverse_call(
        self, y: Array, b: Array, explicit: ExplicitInversePMonLipParams
    ) -> Array:
        return self._explicit_inverse_call_with_diagnostics(y, b, explicit)[0]

    def _explicit_inverse_call_with_diagnostics(
        self, y: Array, b: Array, explicit: ExplicitInversePMonLipParams
    ):
        if b.shape[-1] != sum(self.units):
            raise ValueError(f"b must have {sum(self.units)} features; got {b.shape[-1]}.")
        conditioned = explicit.replace(monlip=explicit.monlip.replace(bh=b))
        return super()._explicit_inverse_call_with_diagnostics(y, conditioned)

    def explicit_call(
        self, params: dict, x: Array, b: Array, explicit: ExplicitPMonLipParams
    ) -> Array:
        """Evaluate the conditioned map using explicit parameters."""
        return self.apply(params, x, b, explicit, method="_explicit_call")

    def inverse_call(
        self, params: dict, y: Array, b: Array, explicit: ExplicitInversePMonLipParams
    ) -> Array:
        """Evaluate the inverse conditioned map using explicit parameters."""
        return self.apply(params, y, b, explicit, method="_explicit_inverse_call")

    def inverse_call_with_diagnostics(
        self, params: dict, y: Array, b: Array, explicit: ExplicitInversePMonLipParams
    ):
        """Evaluate the inverse and return its residual and iteration count."""
        return self.apply(
            params, y, b, explicit, method="_explicit_inverse_call_with_diagnostics"
        )
