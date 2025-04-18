#!/usr/bin/env python3

import jax.numpy as jnp

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array, PrecisionLike

from robustnn.utils import cayley

class Unitary(nn.Module):
    units: int = 0
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        n = jnp.shape(x)[-1]
        m = n if self.units == 0 else self.units
        W = self.param('W',
                       nn.initializers.glorot_normal(),
                       (m, n),
                       jnp.float32)
        a = self.param('a',
                       nn.initializers.constant(jnp.linalg.norm(W)),
                       (1,),
                       jnp.float32)

        R = cayley((a / jnp.linalg.norm(W)) * W)
        z = x @ R.T
        if self.use_bias:
            b = self.param('b', nn.initializers.zeros_init(), (m,), jnp.float32)
            z += b

        return z

    def get_params(self):
        W = self.variables['params']['W']
        a = self.variables['params']['a']
        R = cayley((a / jnp.linalg.norm(W)) * W)
        b = self.variables['params']['b'] if self.use_bias else 0.

        params = {
            'R': R,
            'b': b
        }

        return params
