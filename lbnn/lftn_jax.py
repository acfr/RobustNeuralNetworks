import jax 
import jax.numpy as jnp
from flax import linen as nn 
from typing import Any, Sequence, Callable

eps = 0.05
norms = lambda x: jnp.linalg.norm(x, axis=-1)

def estimate_lipschitz(state, x, dx):
    f = lambda x: state.apply_fn(state.params, x)
    dy = f(x + eps * dx) - f(x - eps * dx)
    lip = 0.5 * norms(dy) / (eps * norms(dx))

    return jnp.max(lip)

def l2_length(x, eps=jnp.finfo(jnp.float32).eps):
    """Compute l2 norm of a vector with JAX."""
    return jnp.sqrt(jnp.maximum(jnp.sum(
        x**2, axis=-1, keepdims=True), eps))

def l2_normalize(x, eps=jnp.finfo(jnp.float32).eps):
    """Normalize x to unit length along last axis.
    This is safe for backpropagation, unlike `jnp.linalg.norm1`."""
    return x / l2_length(x, eps=eps)

def cayley(W):
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    U, V = W[:n, :], W[n:, :]
    Z = (U - U.T) + (V.T @ V)
    I = jnp.eye(n)
    Zi = jnp.linalg.inv(I+Z)

    return jnp.concatenate([Zi @ (I-Z), -2 * V @ Zi], axis=0)

class LipNet(nn.Module):
    units: Sequence[int]
    fout: int 
    gamma: jnp.float32 = 1.0 # Lipschitz bound
    gamma_trainable: bool = False

    # act_fn: Callable = nn.relu
    
    # TODO: This uses the old version of LFTN, not the one in the docs/ folder
    #       We should update this to make the networks more efficient!!

    @nn.compact
    def __call__(self, x : jnp.array) -> jnp.array:
        nx = jnp.shape(x)[-1]  
        ny = self.fout

        gamma = self.param("gamma", nn.initializers.constant(self.gamma), (1,), jnp.float32)

        # If not trainable, update as fixed parameter.
        if not self.gamma_trainable:
            _default_key = jax.random.PRNGKey(0)
            gamma = nn.initializers.constant(self.gamma)(_default_key, (1,), jnp.float32)
        
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (nx+ny, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)
        QT = cayley((fq / jnp.linalg.norm(Fq)) * Fq) 
        
        x = jnp.sqrt(gamma) * x 
        y = 0.
        idx, nz_1 = 0, 0 
        zk = x[..., :0]
        Ak_1 = jnp.zeros((0, 0))
        for k, nz in enumerate(self.units):
            Fab = self.param(f'Fab{k}', nn.initializers.glorot_normal(), (nz+nz_1, nz), jnp.float32)
            fab = self.param(f'fab{k}',nn.initializers.constant(jnp.linalg.norm(Fab)), (1,), jnp.float32)
            ABT = cayley((fab / jnp.linalg.norm(Fab)) * Fab)
            ATk, BTk = ABT[:nz, :], ABT[nz:, :]
            QT_xk_1, QT_xk = QT[:nx, idx-nz_1:idx], QT[:nx, idx:idx+nz]
            QT_yk_1, QT_yk = QT[nx:, idx-nz_1:idx], QT[nx:, idx:idx+nz]
            # use relu activation, no need for psi
            # pk = self.param(f'p{k}', nn.initializers.zeros_init(), (nz,), jnp.float32)
            bk = self.param(f'b{k}', nn.initializers.zeros_init(), (nz,), jnp.float32)
            zk = nn.relu(2 * (zk @ Ak_1 @ BTk + x @ QT_xk @ ATk - x @ QT_xk_1 @ BTk) + bk)
            y += zk @ ATk.T @ QT_yk.T - zk @ BTk.T @ QT_yk_1.T 
            idx += nz 
            nz_1 = nz 
            Ak_1 = ATk.T     

        by = self.param('by', nn.initializers.zeros_init(), (ny,), jnp.float32) 
        y = jnp.sqrt(gamma) * y + by 

        return y 
