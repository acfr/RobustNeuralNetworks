'''
a modified version of monlip and bilip
- instead of taking tau, the program takes mu and nu now
'''

import jax 
import jax.numpy as jnp
from flax import linen as nn 
from typing import Any, Sequence, Callable
from plnet.layer import cayley, Unitary, QuadPotential

# mon-lip-net
class MonLipNet(nn.Module):
    units: Sequence[int]
    mu: jnp.float32 = 0.1 # Monotone lower bound
    nu: jnp.float32 = 10.0 # Lipschitz upper bound (nu > mu)
    # act_fn: Callable = nn.relu

    def get_params(self):
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
            "mu": self.mu,
            "gam": self.nu - self.mu,
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
        by = self.param('by', nn.initializers.zeros_init(), (nx,), jnp.float32) 
        y = self.mu * x + by 
        
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (nx, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)
        QT = cayley((fq / jnp.linalg.norm(Fq)) * Fq) 
        sqrt_2g, sqrt_g2 = jnp.sqrt(2. * (self.nu - self.mu)), jnp.sqrt((self.nu - self.mu) / 2.)
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
    
    def get_bounds(self):
        # todo: check with rui
        return self.mu, self.nu, self.nu/self.mu
    

# bi-lip net
class BiLipNet(nn.Module):
    units: Sequence[int]
    mu: float = 0.1
    nu: float = 10.
    depth: int = 2

    def setup(self):
        uni, mon = [], []
        mu = self.mu ** (1. / self.depth)
        nu = self.nu ** (1. / self.depth)
        for _ in range(self.depth):
            uni.append(Unitary())
            mon.append(MonLipNet(self.units, mu=mu, nu=nu))
        uni.append(Unitary())
        self.uni = uni
        self.mon = mon

    def __call__(self, x: jnp.array) -> jnp.array:
        for k in range(self.depth):
            x = self.uni[k](x)
            x = self.mon[k](x)
        x = self.uni[self.depth](x)
        return x 
    
    def get_bounds(self):
        lipmin, lipmax, tau = 1., 1., 1.
        for k in range(self.depth):
            mu, nu, ta = self.mon[k].get_bounds()
            lipmin *= mu 
            lipmax *= nu 
            tau *= ta 
        return lipmin, lipmax, tau 
    
class PLNet(nn.Module):
    BiLipBlock: nn.Module
    add_constant: float = False

    def gmap(self, x: jnp.array) -> jnp.array:
        return self.BiLipBlock(x)

    def get_bounds(self):
        return self.BiLipBlock.get_bounds()
    
    # plnet result
    def vgap(self, x: jnp.array) -> jnp.array:
        y = self.BiLipBlock(x)
        return 0.5 * (jnp.linalg.norm(y, axis=-1) ** 2)
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        x = self.BiLipBlock(x)
        y = QuadPotential(add_constant = self.add_constant)(x)

        return y 