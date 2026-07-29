# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

'''
Dynamic unitary layer using Cayley transform.

This layer implements a stateful orthogonal/unitary transformation whose stacked
state-space matrix is orthogonal, preserving 2-norms across the transformation.

Author: Yurui Zhang.
'''

import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass
from flax.typing import Array, Dtype

from robustnn.utils import cayley


@dataclass
class DirectDynOrthogonalParams:
    """Data class to keep track of direct params for a dynamic orthogonal layer."""
    X: Array
    bx: Array
    by: Array


@dataclass
class ExplicitDynOrthogonalParams:
    """Data class to keep track of explicit params for a dynamic orthogonal layer."""
    A: Array
    B: Array
    C: Array
    D: Array
    bx: Array
    by: Array


class DynUnitary(nn.Module):
    """Dynamic (stateful) orthogonal layer using a Cayley transform.

    Unlike `Unitary`, this layer is a small *recurrent* linear system whose
    stacked state-space matrix is orthogonal (hence norm-preserving, so it does
    not change the bi-Lipschitz bounds of a network):

        [x1 - bx;  y - by] = G @ [x;  u],     G = [[A, B], [C, D]]  (orthogonal)

    i.e. in row-vector form

        x1 = x @ A.T + u @ B.T + bx     (next state)
        y  = x @ C.T + u @ D.T + by     (output)

    Because `G` is orthogonal, the system is exactly invertible *given the next
    state* `x1` (a non-causal inverse): `[x; u] = G.T @ [x1 - bx;  y - by]`.

    Attributes:
        input_size: input/output feature size (nu == ny).
        state_size: dimension of the internal state.
        use_bias: whether to include the learnable biases bx, by (default: True).
        param_dtype: dtype for the parameters (default: float32).
    """
    input_size: int
    state_size: int
    use_bias: bool = True
    param_dtype: Dtype = jnp.float32

    def setup(self):
        nx, ny = self.state_size, self.input_size
        k = nx + ny
        X = self.param('X', nn.initializers.glorot_normal(), (k, k), self.param_dtype)
        if self.use_bias:
            bx = self.param('bx', nn.initializers.zeros_init(), (nx,), self.param_dtype)
            by = self.param('by', nn.initializers.zeros_init(), (ny,), self.param_dtype)
        else:
            bx = jnp.zeros((nx,), self.param_dtype)
            by = jnp.zeros((ny,), self.param_dtype)
        self.direct = DirectDynOrthogonalParams(X=X, bx=bx, by=by)

    def __call__(self, state: Array, u: Array):
        """Forward pass: (state, inputs) -> (next_state, outputs)."""
        e = self._direct_to_explicit()
        return self._explicit_call(state, u, e)

    def _direct_to_explicit(self) -> ExplicitDynOrthogonalParams:
        nx = self.state_size
        G = cayley(self.direct.X)
        A = G[:nx, :nx]
        B = G[:nx, nx:]
        C = G[nx:, :nx]
        D = G[nx:, nx:]
        return ExplicitDynOrthogonalParams(A, B, C, D, self.direct.bx, self.direct.by)

    def _explicit_call(self, x: Array, u: Array, e: ExplicitDynOrthogonalParams):
        x1 = x @ e.A.T + u @ e.B.T + e.bx
        y = x @ e.C.T + u @ e.D.T + e.by
        return x1, y

    def _explicit_inverse_call(self, x_next: Array, y: Array,
                               e: ExplicitDynOrthogonalParams):
        """Non-causal inverse: recover (prev_state, inputs) from
        (next_state, outputs) using the orthogonality of G."""
        xb = x_next - e.bx
        yb = y - e.by
        x_prev = xb @ e.A + yb @ e.C
        u = xb @ e.B + yb @ e.D
        return x_prev, u

    @nn.nowrap
    def initialize_carry(self, rng, input_shape):
        """Initialise the dynamic-orthogonal state (zeros)."""
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.state_size,)
        return jnp.zeros(mem_shape, self.param_dtype)

    #################### Convenient Wrappers ####################

    def direct_to_explicit(self, params: dict) -> ExplicitDynOrthogonalParams:
        """Convert direct params to explicit params."""
        return self.apply(params, method="_direct_to_explicit")

    def explicit_call(self, params: dict, x: Array, u: Array,
                      explicit: ExplicitDynOrthogonalParams):
        """Evaluate the forward layer given explicit params."""
        return self.apply(params, x, u, explicit, method="_explicit_call")

    def inverse_call(self, params: dict, x_next: Array, y: Array,
                     explicit: ExplicitDynOrthogonalParams):
        """Evaluate the non-causal inverse given explicit params."""
        return self.apply(params, x_next, y, explicit, method="_explicit_inverse_call")
