'''
The layers for plnet
Author: Ruigang Wang
Edited: Dechuan

'''

import jax 
import jax.numpy as jnp
from flax import linen as nn 
from typing import Any, Sequence, Callable

##########################################
# help func
def cayley(W):
    # W in shape n x 2n (m=2n)
    # W = [G H]
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    G, H = W[:n, :], W[n:, :]

    # Z = GT-G + HTH -------- Eq6
    Z = (G - G.T) + (H.T @ H)
    I = jnp.eye(n)
    Zi = jnp.linalg.inv(I+Z)

    # (I+Z)(I-z)-1    -2V(I-Z)-1
    return jnp.concatenate([Zi @ (I-Z), -2 * H @ Zi], axis=0)
##########################################
# layers

# ------------------------------------------------------
# ------------------- all to plnet ---------------------
# ------------------------------------------------------
# o function
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

# non-lip-net
class MonLipNet(nn.Module):
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

# bi-lip net
class BiLipNet(nn.Module):
    units: Sequence[int]
    tau: jnp.float32
    depth: int = 2

    def setup(self):
        uni, mon = [], []
        layer_tau = (self.tau) ** (1/self.depth)
        for _ in range(self.depth):
            uni.append(Unitary())
            mon.append(MonLipNet(self.units, tau=layer_tau))
        uni.append(Unitary())
        self.uni = uni
        self.mon = mon

    def get_bounds(self):
        lipmin, lipmax, tau = 1., 1., 1.
        for k in range(self.depth):
            mu, nu, ta = self.mon[k].get_bounds()
            lipmin *= mu 
            lipmax *= nu 
            tau *= ta 
        return lipmin, lipmax, tau 
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        for k in range(self.depth):
            x = self.uni[k](x)
            x = self.mon[k](x)
        x = self.uni[self.depth](x)
        return x 
    
class PLNet(nn.Module):
    BiLipBlock: nn.Module
    add_constant: float = False #!!!!!

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
        
        y = QuadPotential(add_constant=self.add_constant)(x)
        return y

# ------------------------------------------------------
# ------------------ all to pplnet ---------------------
# ------------------------------------------------------
# partial O
class PUnitary(nn.Module):
    units: int = 0

    @nn.compact
    def __call__(self, x: jnp.array, b: jnp.array) -> jnp.array:
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
        z = x @ R.T + b

        return z 
    
    def get_params(self):
        W = self.variables['params']['W']
        a = self.variables['params']['a']
        R = cayley((a / jnp.linalg.norm(W)) * W)

        params = {
            'R': R
        }

        return params

# partial monotone lip net
class PMonLipNet(nn.Module):
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
        V, S = [], []
        idx = 0
        L = len(self.units)
        for k, nz in zip(range(L), self.units):
            Qk = Q[idx:idx+nz, :] 
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
        # bh = jnp.concatenate(bh, axis=0)
        S = jnp.concatenate(S, axis=0)

        params = {
            "mu": mu,
            "gam": nu - mu,
            "units": self.units,
            "V": V, 
            "S": S,
            "by": by
        }

        return params

    @nn.compact
    def __call__(self, x: jnp.array, b: jnp.array) -> jnp.array:
        nx = jnp.shape(x)[-1]  
        lognu = self.param('lognu', nn.initializers.constant(jnp.log(2.)), (1,), jnp.float32)
        nu = jnp.exp(lognu)
        mu = nu / self.tau 
        by = self.param('by', nn.initializers.zeros_init(), (nx,), jnp.float32) 
        y = mu * x + by 
        
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (nx, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)

        # [eq - 6 above - [Q] = cayley[Fq F*]
        QT = cayley((fq / jnp.linalg.norm(Fq)) * Fq) 

        # sqrt(2* gama), sqrt(gama/2)
        sqrt_2g, sqrt_g2 = jnp.sqrt(2. * (nu - mu)), jnp.sqrt((nu - mu) / 2.)
        idx, nz_1 = 0, 0 
        zk = x[..., :0]
        Ak_1 = jnp.zeros((0, 0))
        for k, nz in enumerate(self.units):
            Fab = self.param(f'Fab{k}', nn.initializers.glorot_normal(), (nz+nz_1, nz), jnp.float32)
            fab = self.param(f'fab{k}',nn.initializers.constant(jnp.linalg.norm(Fab)), (1,), jnp.float32)
            
            # [eq - 6 above - [AT BT] = cayley[Fak Fbk]
            ABT = cayley((fab / jnp.linalg.norm(Fab)) * Fab)
            ATk, BTk = ABT[:nz, :], ABT[nz:, :]

            # [Q1, Q2, ..., QL] [Q1, Q2, ..., Q(L-1)]
            QTk_1, QTk = QT[:, idx-nz_1:idx], QT[:, idx:idx+nz]
            # Eq 22 - S = [A1Q1 A2Q2-B2Q1 ...]
            STk = QTk @ ATk - QTk_1 @ BTk 
            # use relu activation, no need for psi
            # pk = self.param(f'p{k}', nn.initializers.zeros_init(), (nz,), jnp.float32)
            # z = relu (V z + sqrt(2 gama) S x + b) --- eq 24
            zk = nn.relu(2 * (zk @ Ak_1) @ BTk + sqrt_2g * x @ STk + b[..., idx:idx+nz])
            # zk = nn.relu(zk * jnp.exp(-pk)) * jnp.exp(pk)
            # eq 24 - y = (mu x + by) + sqrt(gama/2) S^T z 
            y += sqrt_g2 * zk @ STk.T  
            idx += nz 
            nz_1 = nz 
            Ak_1 = ATk.T     

        return y 

# partially bi-lip net
class PBiLipNet(nn.Module):
    units: Sequence[int]
    po_units: Sequence[int]
    pb_units: Sequence[int]
    tau: jnp.float32
    # mu: float = 0.1
    # nu: float = 10.
    depth: int = 2

    def setup(self):
        uni, mon = [], []
        uni_b, mon_b = [], []
        layer_tau = (self.tau) ** (1/self.depth)
        # mu = self.mu ** (1. / self.depth)
        # nu = self.nu ** (1. / self.depth)
        for _ in range(self.depth):
            uni.append(PUnitary())
            uni_b.append(MLP(self.po_units))
            mon.append(PMonLipNet(self.units, tau=layer_tau))
            mon_b.append(MLP(self.pb_units))
        uni.append(PUnitary())
        uni_b.append(MLP(self.po_units))
        self.uni = uni
        self.mon = mon
        self.uni_b = uni_b
        self.mon_b = mon_b

    def get_bounds(self):
        lipmin, lipmax, tau = 1., 1., 1.
        for k in range(self.depth):
            mu, nu, ta = self.mon[k].get_bounds()
            lipmin *= mu 
            lipmax *= nu 
            tau *= ta 
        return lipmin, lipmax, tau 

    def __call__(self, x: jnp.array, p: jnp.array) -> jnp.array:
        for k in range(self.depth):
            b = self.uni_b[k](p)
            x = self.uni[k](x, b)
            b = self.mon_b[k](p)
            x = self.mon[k](x, b)
        b = self.uni_b[self.depth](p)
        x = self.uni[self.depth](x, b)
        return x

# partial PL net
class PPLNet(nn.Module):
    PBiLipBlock: nn.Module
    add_constant: float = False

    def gmap(self, x: jnp.array, p: jnp.array) -> jnp.array:
        return self.PBiLipBlock(x, p)

    def get_bounds(self):
        return self.PBiLipBlock.get_bounds()
    
    # plnet result
    def vgap(self, x: jnp.array, p: jnp.array) -> jnp.array:
        y = self.PBiLipBlock(x, p)
        return 0.5 * (jnp.linalg.norm(y, axis=-1) ** 2)
    
    @nn.compact
    def __call__(self, x: jnp.array, p: jnp.array) -> jnp.array:
        x = self.PBiLipBlock(x, p)
        y = QuadPotential(add_constant=self.add_constant)(x)

        return jnp.squeeze(y) 


# ------------------------------------------------------
# ------------------- other layers ---------------------
# ------------------------------------------------------
# multi-layer percepton
class MLP(nn.Module):
    features: Sequence[int]

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

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

# linear lip func
class LipLinear(nn.Module):
    unit: int 
    gamma: float = 1.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        n = jnp.shape(x)[-1]
        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (self.unit, n),
                       jnp.float32)
        b = self.param('b', nn.initializers.zeros_init(), (self.unit,), jnp.float32) if self.use_bias else 0.
        x = self.gamma / jnp.linalg.norm(W) * x @ W.T + b
        return x 

# non-linear lip func
class LipNonlin(nn.Module):
    units: Sequence[int] 
    gamma: float = 1.0
    use_bias: bool = True
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        n = jnp.shape(x)[-1]
        for unit in self.units:
            x = LipLinear(unit, gamma=1., use_bias=self.use_bias)(x)
            x = self.act_fn(x)
        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (n, self.units[-1]),
                       jnp.float32)
        x = self.gamma / jnp.linalg.norm(W) * x @ W.T
        return x 
    
class iResNet(nn.Module):
    units: Sequence[int]
    depth: int
    mu: float
    nu: float
    act_fn: Callable = nn.relu

    def setup(self):
        m = (self.mu) ** (1. / self.depth)
        n = (self.nu) ** (1. / self.depth)
        self.a = 0.5 * (m + n)
        self.g = (n - m) / (n + m)

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        for _ in range(self.depth):
            x = self.a * x
            x = x + LipNonlin(self.units, gamma=self.g, act_fn=self.act_fn)(x)

        return x 

class LipSwish(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        beta = self.param('beta', nn.initializers.constant(0.5), (1,), jnp.float32)
        x = x * nn.sigmoid(beta * x) / 1.1
        return x 
                
class iDenseNet(nn.Module):
    units: Sequence[int]
    depth: int 
    mu: float
    nu: float
    use_lipswich: bool = False
    

    def setup(self):
        m = (self.mu) ** (1. / self.depth)
        n = (self.nu) ** (1. / self.depth)
        self.a = 0.5 * (m + n)
        self.g = (n - m) / (n + m)
        if self.use_lipswich:
            self.act_fn = LipSwish()
        else:
            self.act_fn = nn.relu

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        for _ in range(self.depth):
            x = self.a * x 
            x = x + LipNonlin(self.units, gamma=self.g, act_fn=self.act_fn)(x)

        return x 

class QuadPotential(nn.Module):
    add_constant: float = False
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        
        y = 0.5 * jnp.mean(jnp.square(x), axis=-1)
        # y = 0.5 * (jnp.linalg.norm(x, axis=-1) ** 2) 
        if self.add_constant:
            y += self.param('c', nn.initializers.constant(0.), (1,), jnp.float32)
        return y

class SquarePotential(nn.Module):
    add_constant: float = False
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        
        y = jnp.max(jnp.abs(x), axis=-1)
        # y = 0.5 * (jnp.linalg.norm(x, axis=-1) ** 2) 
        if self.add_constant:
            y += self.param('c', nn.initializers.constant(0.), (1,), jnp.float32)
        return y

# use plnet to calculate the differetnce between optimal and input x
# 0.5*|g(x)-g(x_opt)|^2
class V_PLNet(nn.Module):
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
    def __call__(self, x: jnp.array, x_optimal: jnp.array) -> jnp.array:
        gv_x = self.BiLipBlock(x)
        gv_x_optimal = self.BiLipBlock(x_optimal)
        y = QuadPotential(add_constant = self.add_constant)(gv_x - gv_x_optimal)

        return y 


# add class for J-R is HMM model
class JRModel(nn.Module):
    size: int  # Dimension of the square matrices J and R
    hidden_layers: Sequence[int]  # Number of units in each hidden layer for the MLPs
    epislon: jnp.float32 = 0.0   # eposlon term added to R to ensure decrease in H (energy)

    def setup(self):
        # Define MLPs for transforming input x to produce J and R
        self.j_mlp = nn.Sequential([
            layer for hidden_dim in self.hidden_layers
            for layer in (nn.Dense(hidden_dim), nn.relu)
        ] + [nn.Dense(self.size * self.size)])  # Output size * size for reshaping into a matrix

        self.r_mlp = nn.Sequential([
            layer for hidden_dim in self.hidden_layers
            for layer in (nn.Dense(hidden_dim), nn.relu)
        ] + [nn.Dense(self.size * self.size)])  # Output size * size for reshaping into a matrix

    def _make_skew_symmetric(self, matrix):
        """Enforce skew-symmetry: J = -J^T."""
        return matrix - jnp.transpose(matrix, (0, 2, 1)) 

    def _make_positive_definite(self, matrix):
        sym_matrix = matrix @ jnp.transpose(matrix, (0, 2, 1)) + self.epislon * jnp.eye(self.size)
        return sym_matrix

    def __call__(self, x):
        # Generate flat representations of J and R using the MLPs
        j_flat = self.j_mlp(x)  # Output shape: (size * size,)
        r_flat = self.r_mlp(x)  # Output shape: (size * size,)

        # Reshape them into square matrices
        J_matrix = jnp.reshape(j_flat, (x.shape[0], self.size, self.size))
        R_matrix = jnp.reshape(r_flat, (x.shape[0], self.size, self.size))

        # Apply transformations to enforce desired properties
        J = self._make_skew_symmetric(J_matrix)
        R = self._make_positive_definite(R_matrix)

        # jax.debug.print("J-R: {} ", jax.numpy.shape(J-R))

        return J - R
    
# Define the Hamiltonian Neural Network (HNN) model
class HNN(nn.Module):
    jr_model: JRModel      # Instance of JRModel that computes (J - R) matrix operation
    h_net: Callable        # Callable representing PLNet to approximate the Hamiltonian function H (call by h_net(point, equillibrium_point))
    trainable_h_net: bool  # If True, h_net is learnable; if False, h_net is frozen
    equillibrium_point: jax.Array # give the equillibrium point for H
    norm_cap_H:float = jax.numpy.inf # use this value to limit the maximum value of norm(grad(H))

    @nn.compact
    def __call__(self, x):
        # Calculate the gradient of H with respect to x.
        # If `trainable_h_net` is False, this won't backpropagate to h_net parameters.
        grad_H = jax.grad(lambda x: self.h_net(x, self.equillibrium_point))

        def norm_cap_grad_H(point):
            return jnp.where(grad_H(point)<self.norm_cap_H, 
                    grad_H(point), 
                    point / jnp.linalg.norm(point) * self.norm_cap_H)

        # grad_H = jax.grad(lambda x: self.h_net(x) if self.trainable_h_net else jax.lax.stop_gradient(self.h_net(x)))
        grad_H = jax.vmap(norm_cap_grad_H)(x)

        grad_H = grad_H if self.trainable_h_net else jax.lax.stop_gradient(grad_H)

        # Apply the (J - R) matrix operation from JRModel on the gradient of H.
        # jax.debug.print("output from J-R: {}", self.jr_model(x))
        # x_dot = jnp.einsum('ijk,ik->ik', self.jr_model(x), grad_H)
        # jax.debug.print("grad shape: {} and x_dot: {}", jax.numpy.shape(grad_H), jnp.shape(x_dot))

        # x_dot = jnp.einsum('ijk,ikl->ijl', self.jr_model(x), jnp.expand_dims(grad_H, axis=-1))
        x_dot = jnp.matmul(self.jr_model(x), jnp.expand_dims(grad_H, axis=-1))
        # jax.debug.print("grad shape: {} and x_dot: {}", jax.numpy.shape(jnp.expand_dims(grad_H, axis=-1)), jnp.shape(x_dot))
        x_dot =  jnp.squeeze(x_dot, axis=-1)

        return x_dot

    def get_H_value(self, x):
        return self.h_net(x, self.equillibrium_point)

    def get_grad_H_value(self, x):
        grad_H = jax.grad(lambda x: self.h_net(x, self.equillibrium_point))

        return jax.vmap(grad_H)(x)
    
    def get_bounds(self):
        if self.trainable_h_net:
            return self.h_net.get_bounds()
        else:
            return 0., 0., 0.

    def get_JR(self, x):
        return self.jr_model(x)

    def gmap(self, x: jnp.array) -> jnp.array:
        return self.h_net.gmap(x)
