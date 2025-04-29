import jax.numpy as jnp
from flax import linen as nn 
from typing import Any, Sequence, Callable
from flax.typing import Array, PrecisionLike
from robustnn.utils import cayley
from flax.struct import dataclass

@dataclass
class ImplicitMonLipParams:
    """Data class to keep track of implicit params for Monontone Lipschitz layer."""
    mu: Array
    nu: Array
    Fq: Array
    fq: Array
    by: Array
    b: Array
    Fab: Array
    fab: Array

@dataclass
class ExplicitMonLipParams:
    """Data class to keep track of explicit params for Monontone Lipschitz layer."""
    mu: Array
    nu: Array
    units: Sequence[int]
    V: Array
    S: Array
    by: Array
    bh: Array
class MonLipNet(nn.Module):
    '''
    Monotone Lipschitz neural network layer using Cayley transform.
    This layer applies a learned monotone Lipschitz transformation to the input
    using the Cayley map, preserving 2-norms in the transformation process.
    Example usage::

        >>> layer = MonLipNet(units=[4, 4])
        >>> x = jnp.ones((1, 4))
        >>> params = layer.init(jax.random.key(0), x)
        >>> y = layer.apply(params, x)
    Attributes:
        input_size: Size of the input features.
        units: Sequence of integers representing the number of output features for each layer.
        tau: Scaling factor for distortion (default: 10.0).
        mu: Monotone lower bound (default: 0.1).
        nu: Lipschitz upper bound (default: 10.0).
        is_mu_fixed: Whether to fix the value of mu (default: False).
        is_nu_fixed: Whether to fix the value of nu (default: False).
        is_tau_fixed: Whether to fix the value of tau (default: False). Note that you cannot have 
            is_tau_fixed, is_mu_fixed, and is_nu_fixed at the same time.
        act_fn: Activation function to be used in the layer (default: nn.relu).
    '''
    input_size: int
    units: Sequence[int]
    tau: jnp.float32 = 10.
    mu: jnp.float32 = 0.1 # Monotone lower bound
    nu: jnp.float32 = 10.0 # Lipschitz upper bound (nu > mu)
    is_mu_fixed: bool = False
    is_nu_fixed: bool = False
    is_tau_fixed: bool = False
    act_fn: Callable = nn.relu

    def setup(self):
        """Setup method for the MonLipNet layer."""

        # setup mu, nu, tau
        if self.is_mu_fixed and self.is_nu_fixed and self.is_tau_fixed:
            raise ValueError("Cannot fix mu, nu, and tau at the same time.")
        elif self.is_mu_fixed and self.is_nu_fixed:
            mu = self.mu
            nu = self.nu
        elif self.is_mu_fixed and self.is_tau_fixed:
            mu = self.mu
            nu = self.tau * self.mu
        elif self.is_nu_fixed and self.is_tau_fixed:
            nu = self.nu
            mu = self.nu / self.tau
        elif self.is_mu_fixed:
            mu = self.mu
            log_nu = self.param('lognu', nn.initializers.constant(jnp.log(self.nu)), (1,), jnp.float32)
            nu = jnp.exp(log_nu)
        elif self.is_nu_fixed:
            nu = self.nu
            log_mu = self.param('logmu', nn.initializers.constant(jnp.log(self.mu)), (1,), jnp.float32)
            mu = jnp.exp(log_mu)
        elif self.is_tau_fixed:
            tau = self.tau
            log_mu = self.param('logmu', nn.initializers.constant(jnp.log(self.mu)), (1,), jnp.float32)
            mu = jnp.exp(log_mu)
            nu = tau * mu
        else:
            log_mu = self.param('logmu', nn.initializers.constant(jnp.log(self.mu)), (1,), jnp.float32)
            mu = jnp.exp(log_mu)
            log_nu = self.param('lognu', nn.initializers.constant(jnp.log(self.nu)), (1,), jnp.float32)
            nu = jnp.exp(log_nu)

        by = self.param('by', nn.initializers.zeros_init(), (self.input_size,), jnp.float32)
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (self.input_size, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)
        QT = cayley((fq / jnp.linalg.norm(Fq)) * Fq)



class MonLipNet_old(nn.Module):
    units: Sequence[int]
    tau: jnp.float32 = 10.
    # mu: jnp.float32 = 0.1 # Monotone lower bound
    # nu: jnp.float32 = 10.0 # Lipschitz upper bound (nu > mu)
    # act_fn: Callable = nn.relu

    def get_bounds(self):
        lognu = self.variables['params']['lognu']
        nu = jnp.squeeze(jnp.exp(lognu), 0)
        mu = nu / self.tau

        return mu, nu, self.tau
    
    def get_params(self):
        lognu = self.variables['params']['lognu']
        nu = jnp.squeeze(jnp.exp(lognu), 0)
        mu = nu / self.tau
        Fq = self.variables['params']['Fq']
        fq = self.variables['params']['fq']
        Q = cayley((fq / jnp.linalg.norm(Fq)) * Fq).T 
        V, S, bh = [], [], []
        idx = 0
        L = len(self.units)
        for k, nz in zip(range(L), self.units):
            Qk = Q[idx:idx+nz, :] 
            b = self.variables['params'][f'b{k}']
            bh.append(b)
            Fab = self.variables['params'][f'Fab{k}']
            fab = self.variables['params'][f'fab{k}']
            ABT = cayley((fab / jnp.linalg.norm(Fab)) * Fab)
            if k > 0:
                Ak, Bk = ABT[:nz, :].T, ABT[nz:, :].T
                V.append(2 * Bk @ ATk_1)
                S.append(Ak @ Qk - Bk @ Qk_1)
            else:
                Ak = ABT.T      
                S.append(ABT.T @ Qk)
            
            ATk_1, Qk_1 = Ak.T, Qk
            idx += nz

        by = self.variables['params']['by']
        bh = jnp.concatenate(bh, axis=0)
        S = jnp.concatenate(S, axis=0)

        params = {
            "mu": mu,
            "gam": nu - mu,
            "units": self.units,
            "V": V, 
            "S": S,
            "by": by,
            "bh": bh
        }

        return params

    @nn.compact
    def __call__(self, x : jnp.array) -> jnp.array:
        nx = jnp.shape(x)[-1]  
        lognu = self.param('lognu', nn.initializers.constant(jnp.log(2.)), (1,), jnp.float32)
        nu = jnp.exp(lognu)
        mu = nu / self.tau 
        by = self.param('by', nn.initializers.zeros_init(), (nx,), jnp.float32) 
        y = mu * x + by 
        
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (nx, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)
        QT = cayley((fq / jnp.linalg.norm(Fq)) * Fq) 
        sqrt_2g, sqrt_g2 = jnp.sqrt(2. * (nu - mu)), jnp.sqrt((nu - mu) / 2.)

        idx, nz_1 = 0, 0 
        zk = x[..., :0]
        Ak_1 = jnp.zeros((0, 0))
        for k, nz in enumerate(self.units):
            Fab = self.param(f'Fab{k}', nn.initializers.glorot_normal(), (nz+nz_1, nz), jnp.float32)
            fab = self.param(f'fab{k}',nn.initializers.constant(jnp.linalg.norm(Fab)), (1,), jnp.float32)
            ABT = cayley((fab / jnp.linalg.norm(Fab)) * Fab)
            ATk, BTk = ABT[:nz, :], ABT[nz:, :]
            QTk_1, QTk = QT[:, idx-nz_1:idx], QT[:, idx:idx+nz]
            STk = QTk @ ATk - QTk_1 @ BTk 
            bk = self.param(f'b{k}', nn.initializers.zeros_init(), (nz,), jnp.float32)
            # use relu activation, no need for psi
            # pk = self.param(f'p{k}', nn.initializers.zeros_init(), (nz,), jnp.float32)
            zk = nn.relu(2 * (zk @ Ak_1) @ BTk + sqrt_2g * x @ STk + bk)
            # zk = nn.relu(zk * jnp.exp(-pk)) * jnp.exp(pk)
            y += sqrt_g2 * zk @ STk.T  
            idx += nz 
            nz_1 = nz 
            Ak_1 = ATk.T     

        return y 
    
    