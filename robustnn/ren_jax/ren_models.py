import jax
import jax.numpy as jnp
from robustnn.ren_jax.ren_base import RENBase, DirectRENParams, ExplicitRENParams

from typing import Union, Any, Tuple

Array = Union[jax.Array, Any]

class ContractingREN(RENBase):
    """Construct a Contracting REN.
    
    Example usage::

        >>> from robustnn.ren_jax.ren_models import ContractingREN
        >>> import jax, jax.numpy as jnp
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, nv, ny = 1, 2, 4, 1
        >>> ren = ContractingREN(nu, nx, nv, ny)
        
        >>> batches = 5
        >>> states = ren.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = ren.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 1), 'X': (8, 8), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bv': (4,), 'bx': (2,), 'by': (1,), 'polar': (1,)}}
    
    See docs for `RENBase` for full list of arguments.
    """
    d22_free: bool = True
    
    def setup(self):
        if not self.d22_free:
            raise ValueError("Set `d22_free=True` for contracting RENs.")
    
    def direct_to_explicit(self, ps: DirectRENParams) -> ExplicitRENParams:
        H = self.x_to_h(ps.X, ps.p)
        explicit = self.hmatrix_to_explicit(ps, H, ps.D22)
        return explicit
    
    
class LipschitzREN(RENBase):
    """Construct a Lipschitz-bounded REN.
    
    Attributes::
        gamma: upper bound on the Lipschitz constant (default 1.0).
    
    Example usage::

        >>> from robustnn.ren_jax.ren_models import LipschitzREN
        >>> import jax, jax.numpy as jnp
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, nv, ny = 1, 2, 4, 1
        >>> ren = LipschitzREN(nu, nx, nv, ny, gamma=10.0)
        
        >>> batches = 5
        >>> states = ren.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = ren.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 1), 'X': (8, 8), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bv': (1, 4), 'bx': (1, 2), 'by': (1, 1), 'polar': (1,)}}
    
    See docs for `RENBase` for full list of arguments.
    """
    gamma: jnp.float32 = 1.0
    
    def setup(self):
        if self.d22_free:
            raise ValueError("Set `d22_free=False` for Lipschitz RENs.")
    
    def direct_to_explicit(self, ps: DirectRENParams) -> ExplicitRENParams:
        nu = self.input_size
        nx = self.state_size
        ny = self.output_size
        
        # Implicit params
        B2_imp = ps.B2
        D12_imp = ps.D12
        
        # Construct D22 (Eqns 31-33 of Revay et al. (2023))
        if self.d22_zero:
            D22 = ps.D22
        else:
            I = jnp.identity(nu, self.param_dtype)
            M = ps.X3.T @ ps.X3 + ps.Y3 - ps.Y3.T + ps.Z3.T @ ps.Z3 + self.eps*I
            if ny >= nu:
                N = jnp.vstack((jnp.linalg.solve((I + M).T, (I - M).T).T,
                                jnp.linalg.solve((I + M).T, -2*ps.Z3.T).T))
            else:
                N = jnp.hstack((jnp.linalg.solve((I + M), (I - M)),
                                jnp.linalg.solve((I + M), -2*ps.Z3.T)))
            D22 = self.gamma * N
        
        # Construct H (Eqn. 28 of Revay et al. (2023))
        C2_imp = -D22.T @ ps.C2 / self.gamma
        D21_imp = -(D22.T @ ps.D21) / self.gamma - D12_imp.T
        
        R = self.gamma * (-D22.T @ D22 / (self.gamma**2) + I)
        mul_Q = jnp.hstack((ps.C2, ps.D21, jnp.zeros((ny, nx), self.param_dtype)))
        mul_R = jnp.hstack((C2_imp, D21_imp, B2_imp.T))
        Gamma_Q = mul_Q.T @ mul_Q / (-self.gamma)
        Gamma_R = mul_R.T @ jnp.linalg.solve(R, mul_R)
        
        H = self.x_to_h(ps.X, ps.p) + Gamma_R - Gamma_Q
        explicit = self.hmatrix_to_explicit(ps, H, D22)
        return explicit

class GeneralREN(RENBase):
    """Construct a REN satisfying an incremental IQC defined by Q, S, R.
    
    Example usage::

        >>> from robustnn.ren_jax.ren_models import GeneralREN
        >>> import jax, jax.numpy as jnp
        
        >>> rng = jax.random.key(0)
        >>> rng, keyX, keyY, keyS, key1, key2 = jax.random.split(rng, 6)

        >>> # Set up some IQC paramters for testing
        >>> nu, nx, nv, ny = 1, 2, 4, 1
        >>> X = jax.random.normal(keyX, (ny, ny))
        >>> Y = jax.random.normal(keyY, (nu, nu))
        >>> S = jax.random.normal(keyS, (nu, ny))
        >>> Q = -X.T @ X
        >>> R = S @ jnp.linalg.solve(Q, S.T) + Y.T @ Y
        
        >>> # Construct REN and check for valid IQC params
        >>> ren = LipschitzREN(nu, nx, nv, ny, gamma=10.0)
        >>> ren.check_valid_qsr(*ren.qsr)
        
        >>> batches = 5
        >>> states = ren.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = ren.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
    
    Attributes::
        qsr: tuple of IQC matrices (Q, S, R), see Eqn. (5) of Revay et al. (2023).
    
    The IQC matrices have the following conditions for a REN with input
    size `nu` and output size `ny`:
        - `Q.shape` must be `(ny, ny)`.
        - `S.shape` must be `(nu, ny)`.
        - `R.shape` must be `(nu, nu)`.
        - `Q` must be negative definite.
        - `R - S @ (inv(Q) @ S.T)` must be positive definite.
    We expect users to JIT calls to the `.init()` and `.apply()` methods for a
    REN, so we can't do any error checking to make sure the symmetric/negative
    semi-definite conditions are met. Instead, we leave this up to the user
    and provide tools to check for appropriate (Q, S, R) matrices (see above).
    """
    qsr: Tuple[Array, Array, Array] = (None, None, None)
    
    def setup(self):
        if self.d22_free:
            raise ValueError("Set `d22_free=False` for general QSR RENs")
        if self.d22_zero:
            raise ValueError("Cannot have zero D22 for general QSR REN.")
        Q, self.S, R = self.qsr
        
        # Small delta to help numerical conditioning with cholesky decomposition
        self.Q = Q - self.eps * jnp.identity(Q.shape[0], Q.dtype)
        self.R = R + self.eps * jnp.identity(R.shape[0], R.dtype)
        
    
    def direct_to_explicit(self, ps: DirectRENParams) -> ExplicitRENParams:
        nu = self.input_size
        nx = self.state_size
        ny = self.output_size
        
        # Compute useful decompositions
        R_temp = self.R - self.S @ jnp.linalg.solve(self.Q, self.S.T)
        LQ = jnp.linalg.cholesky(-self.Q, upper=True)
        LR = jnp.linalg.cholesky(R_temp, upper=True)
        
        # Implicit params
        B2_imp = ps.B2
        D12_imp = ps.D12
        
        # Construct D22 (Eqns 31-33 of Revay et al. (2023))
        I = jnp.identity(nu, self.param_dtype)
        M = ps.X3.T @ ps.X3 + ps.Y3 - ps.Y3.T + ps.Z3.T @ ps.Z3 + self.eps*I
        if ny >= nu:
            N = jnp.vstack((jnp.linalg.solve((I + M).T, (I - M).T).T,
                            jnp.linalg.solve((I + M).T, -2*ps.Z3.T).T))
        else:
            N = jnp.hstack((jnp.linalg.solve((I + M), (I - M)),
                            jnp.linalg.solve((I + M), -2*ps.Z3.T)))
        
        D22 = jnp.linalg.solve(-self.Q, self.S.T) + jnp.linalg.solve(LQ, N) @ LR
        
        # Construct H (Eqn. 28 of Revay et al. (2023))
        C2_imp = (D22.T @ self.Q + self.S) @ ps.C2
        D21_imp = (D22.T @ self.Q + self.S) @ ps.D21 - D12_imp.T
        
        R1 = self.R + self.S @ D22 + D22.T @ self.S.T + D22.T @ self.Q @ D22
        mul_Q = jnp.hstack((ps.C2, ps.D21, jnp.zeros((ny, nx), self.param_dtype)))
        mul_R = jnp.hstack((C2_imp, D21_imp, B2_imp.T))
        Gamma_Q = mul_Q.T @ self.Q @ mul_Q
        Gamma_R = mul_R.T @ jnp.linalg.solve(R1, mul_R)
        
        H = self.x_to_h(ps.X, ps.p) + Gamma_R - Gamma_Q
        explicit = self.hmatrix_to_explicit(ps, H, D22)
        return explicit

    def check_valid_qsr(self, Q, S, R):
        """Check that the (Q,S,R) matrices are valid.
        
        Example usage:
            >>> Q, S, R = ... # Define your matrices here.
            
            >>> nu, nx, nv, ny = 1, 3, 4, 2
            >>> ren = GeneralREN(nu, nx, nv, ny, qsr=(Q, S, R))
            >>> ren.check_valid_qsr(*self.qsr)
        """
        nu = self.input_size
        ny = self.output_size
        
        if not Q.shape == (ny, ny):
            raise ValueError("`Q` should have size `(output_size, output_size)`.")
        
        if not S.shape == (nu, ny):
            raise ValueError("`S` should have size `(input_size, output_size)`.")
        
        if not R.shape == (nu, nu):
            raise ValueError("`R` should have size `(input_size, input_size)`.")
        
        if not _check_posdef(-Q):
            raise ValueError("`Q` must be negative definite.")
        
        R_temp = R - S @ jnp.linalg.solve(Q, S.T)
        if not _check_posdef(R_temp):
            raise ValueError("`R - S @ (inv(Q) @ S.T)` must be positive definite.")


def _check_posdef(A: Array, eps=jnp.finfo(jnp.float32).eps):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        return False
    A = A + eps * jnp.identity(A.shape[0], A.dtype)
    if not jnp.allclose(A, A.T):
        return False
    if not jnp.all(jnp.linalg.eigh(A)[0] > 0.0):
        return False
    return True
