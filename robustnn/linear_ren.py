# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Base class for linear Recurrent Equilibrium Networks (RENs).

See ./robustnn/ren.py for a description. These linear RENs have nv = 0.

Author: Nic Barbara.
'''

import jax.numpy as jnp
from flax.typing import Array
from robustnn import linear_ren_base as ren
from robustnn.ren import _check_valid_qsr, _adjust_iqc_params

class ContractingLinREN(ren.LinRENBase):
    """Construct a contracting linear REN.

    Example usage:

        >>> import jax, jax.numpy as jnp
        >>> from robustnn import linear_ren as ren
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, ny = 1, 2, 1
        >>> model = ren.ContractingLinREN(nu, nx, 0, ny)
        
        >>> batches = 5
        >>> states = model.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = model.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B': (2, 1), 'C': (1, 2), 'D': (1, 1), 'X': (4, 4), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bx': (2,), 'by': (1,), 'p': (1,)}}
    
    See docs for `LinRENBase` for full list of arguments.
    """
    
    def _direct_to_explicit(self) -> ren.ExplicitLinRENParams:
        ps = self.direct
        H = self._x_to_h_contracting(ps.X, ps.p)
        explicit = self._hmatrix_to_explicit(ps, H, ps.D)
        return explicit


class LipschitzLinREN(ren.LinRENBase):
    """Construct a Lipschitz-bounded linear REN.
    
    Attributes:
        gamma: upper bound on the Lipschitz constant (default 1.0).
    
    Example usage:

        >>> import jax, jax.numpy as jnp
        >>> from robustnn import linear_ren as ren
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, ny = 1, 2, 1
        >>> model = ren.LipschitzLinREN(nu, nx, 0, ny, gamma=10.0)
        
        >>> batches = 5
        >>> states = model.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = model.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B': (2, 1), 'C': (1, 2), 'D': (1, 1), 'X': (4, 4), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bx': (2,), 'by': (1,), 'p': (1,)}}
    
    See docs for `LinRENBase` for full list of arguments.
    """
    gamma: jnp.float32 = 1.0 # type: ignore
    
    def _direct_to_explicit(self) -> ren.ExplicitLinRENParams:
        ps = self.direct
        nu = self.input_size
        nx = self.state_size
        ny = self.output_size
        dtype = self.param_dtype
        
        Iu = jnp.identity(nu, dtype)
        Iy = jnp.identity(ny, dtype)
        
        # Construct D (Eqns 31-33 of Revay et al. (2023))
        if self.d22_zero:
            D = ps.D
        else:
            M = ps.X3.T @ ps.X3 + ps.Y3 - ps.Y3.T + ps.Z3.T @ ps.Z3 + self.eps*Iy
            if ny >= nu:
                N = jnp.vstack((jnp.linalg.solve((Iy + M).T, (Iy - M).T).T,
                                jnp.linalg.solve((Iy + M).T, -2*ps.Z3.T).T))
            else:
                N = jnp.hstack((jnp.linalg.solve((Iy + M), (Iy - M)),
                                jnp.linalg.solve((Iy + M), -2*ps.Z3.T)))
            D = self.gamma * N
        
        # Construct H (Eqn. 28 of Revay et al. (2023))
        B_imp = ps.B
        C_imp = -D.T @ ps.C / self.gamma
        R = self.gamma * (-D.T @ D / (self.gamma**2) + Iu)
        
        mul_R = jnp.hstack((C_imp, B_imp.T))
        Gamma_R = mul_R.T @ jnp.linalg.solve(R, mul_R)

        zeros_x = jnp.zeros((nx, nx), dtype)
        Gamma_Q = jnp.block([[ps.C.T @ ps.C, zeros_x], [zeros_x, zeros_x]]) / self.gamma
        
        H = self._x_to_h_contracting(ps.X, ps.p) + Gamma_R + Gamma_Q
        explicit = self._hmatrix_to_explicit(ps, H, D)
        return explicit


class GeneralLinREN(ren.LinRENBase):
    """Construct a linear REN satisfying an incremental IQC defined by Q, S, R.
    
    Example usage:

        >>> import jax, jax.numpy as jnp
        >>> from robustnn import linear_ren as ren
        
        >>> rng = jax.random.key(0)
        >>> rng, keyX, keyY, keyS, key1, key2 = jax.random.split(rng, 6)

        >>> # Set up some IQC paramters for testing
        >>> nu, nx, ny = 1, 2, 1
        >>> X = jax.random.normal(keyX, (ny, ny))
        >>> Y = jax.random.normal(keyY, (nu, nu))
        >>> S = jax.random.normal(keyS, (nu, ny))
        >>> Q = -X.T @ X
        >>> R = S @ jnp.linalg.solve(Q, S.T) + Y.T @ Y
        
        >>> # Construct REN and check for valid IQC params
        >>> model = ren.GeneralLinREN(nu, nx, 0, ny, Q=Q, S=S, R=R)
        >>> model.check_valid_qsr()
        
        >>> batches = 5
        >>> states = model.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = model.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B': (2, 1), 'C': (1, 2), 'D': (1, 1), 'X': (4, 4), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bx': (2,), 'by': (1,), 'p': (1,)}}
        
    Attributes:
        Q: IQC output weight.
        S: IQC cross input/output weight.
        R: IQC input weight.
    
    The IQC matrices have the following conditions for a linear REN with input
    size `nu` and output size `ny`:
    
    - `Q.shape` must be `(ny, ny)`.
    - `S.shape` must be `(nu, ny)`.
    - `R.shape` must be `(nu, nu)`.
    - `Q` must be negative definite.
    - `R - S @ (inv(Q) @ S.T)` must be positive definite.
    
    We expect users to JIT calls to the `.init()` and `.apply()` methods for a
    REN, so we leave error checking as a separate API call. Use 
    `model.check_valid_qsr()` to check for appropriate (Q, S, R) matrices.
    """
    Q: Array = None
    S: Array = None
    R: Array = None
    
    def _error_checking(self):
        if (not self.d22_zero) and self.init_output_zero:
            raise ValueError(
                "Cannot have zero output on init without setting `d22_zero=True`."
            )
            
    def _direct_to_explicit(self) -> ren.ExplicitLinRENParams:
        ps = self.direct
        nu = self.input_size
        nx = self.state_size
        ny = self.output_size
        Q, S, R = _adjust_iqc_params(self.Q, self.S, self.R, self.eps, self.param_dtype)

        # Compute useful decompositions
        R_temp = R - S @ jnp.linalg.solve(Q, S.T)
        LQ = jnp.linalg.cholesky(-Q, upper=True)
        LR = jnp.linalg.cholesky(R_temp, upper=True)
        
        # Construct D (Eqns 31-33 of Revay et al. (2023))
        if self.d22_zero:
            D = ps.D
        else:
            I = jnp.identity(ny, self.param_dtype)
            M = ps.X3.T @ ps.X3 + ps.Y3 - ps.Y3.T + ps.Z3.T @ ps.Z3 + self.eps*I
            if ny >= nu:
                N = jnp.vstack((jnp.linalg.solve((I + M).T, (I - M).T).T,
                                jnp.linalg.solve((I + M).T, -2*ps.Z3.T).T))
            else:
                N = jnp.hstack((jnp.linalg.solve((I + M), (I - M)),
                                jnp.linalg.solve((I + M), -2*ps.Z3.T)))
            
            D = jnp.linalg.solve(-Q, S.T) + jnp.linalg.solve(LQ, N) @ LR
        
        # Construct H (Eqn. 28 of Revay et al. (2023))
        B_imp = ps.B
        C_imp = (D.T @ Q + S) @ ps.C
        
        R1 = R + S @ D + D.T @ S.T + D.T @ Q @ D
        mul_Q = jnp.hstack((ps.C, jnp.zeros((ny, nx), self.param_dtype)))
        mul_R = jnp.hstack((C_imp, B_imp.T))
        
        Gamma_Q = mul_Q.T @ Q @ mul_Q
        Gamma_R = mul_R.T @ jnp.linalg.solve(R1, mul_R)
        
        H = self._x_to_h_contracting(ps.X, ps.p) + Gamma_R - Gamma_Q
        explicit = self._hmatrix_to_explicit(ps, H, D)
        return explicit
        
    def check_valid_qsr(self):
        """Check that the (Q,S,R) matrices are valid.
        
        Example usage:
            >>> Q, S, R = ... # Define your matrices here.
            
            >>> nu, nx, nv, ny = 1, 3, 2
            >>> ren = GeneralLinREN(nu, nx, 0, ny, Q=Q, S=S, R=R)
            >>> ren.check_valid_qsr()
            
        This function is NOT run automatically in the `setup()` routine
        to avoid issues with the JAX tracer.
        """
        nu = self.input_size
        ny = self.output_size
        _check_valid_qsr(nu, ny, self.Q, self.S, self.R, self.eps, self.param_dtype)