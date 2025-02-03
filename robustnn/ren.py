import jax.numpy as jnp

from flax.linen import initializers as init
from flax.typing import Array
from typing import Tuple

from robustnn.utils import l2_norm, identity_init, solve_discrete_lyapunov_direct
from robustnn.ren_base import RENBase, DirectRENParams, ExplicitRENParams

class ContractingREN(RENBase):
    """Construct a Contracting REN.
    
    Attributes::
        init_as_linear: Tuple of (A, B, C, D) matrices to initialise the contracting REN as
                        a linear system. The linear system must be stable. Default is `()`, 
                        which uses the random initialisation outlined in `RENBase`.
    
    Example usage::

        >>> import jax, jax.numpy as jnp
        >>> from robustnn import ren
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, nv, ny = 1, 2, 4, 1
        >>> model = ren.ContractingREN(nu, nx, nv, ny)
        
        >>> batches = 5
        >>> states = model.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = model.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 1), 'X': (8, 8), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bv': (4,), 'bx': (2,), 'by': (1,), 'polar': (1,)}}
    
    See docs for `RENBase` for full list of arguments.
    """
    d22_free: bool = True
    init_as_linear: Tuple = ()
    
    def setup(self):
        self._error_checking()
        if not self.init_as_linear:
            self._init_params()
        else:
            self._init_linear_sys()
    
    def _error_checking(self):
        if not self.d22_free:
            raise ValueError("Set `d22_free=True` for contracting RENs.")
        
    def _init_linear_sys(self):
        """Initialise the contracting REN as a (stable) linear system."""
        
        # Extract params and system sizes
        A, B, C, D = self.init_as_linear
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        
        # Error checking
        nx_a = A.shape[0]
        if not (A.shape[0] <= nx and A.shape[1] == A.shape[0]):
            raise ValueError("Size of initial A matrix should be `(n, n)` with `n <= state_size`")
        if not (B.shape[0] == nx_a and B.shape[1] == nu):
            raise ValueError("Size of initial B matrix should be `(A.shape[0], input_size)`")
        if not (C.shape[0] == ny and C.shape[1] == nx_a):
            raise ValueError("Size of initial C matrix should be `(output_size, A.shape[0])`")
        if not (D.shape[0] == ny and D.shape[1] == nu):
            raise ValueError("Size of initial D matrix should be `(output_size, input_size)`") 
        if not self.abar == 1:
            raise NotImplementedError("Make compatible with abar != 0 (TODO).")
        
        # Fill out A matrix to match the required number of states
        dnx = nx - nx_a
        A = jnp.block([
            [A, jnp.zeros((nx_a, dnx), self.param_dtype)],
            [jnp.zeros((dnx, nx_a), self.param_dtype), jnp.zeros((dnx, dnx), self.param_dtype)],
        ])
        B = jnp.vstack([B, jnp.zeros((dnx, nu), self.param_dtype)])
        C = jnp.hstack([C, jnp.zeros((ny, dnx), self.param_dtype)])
        
        # Make sure all the biases are zero
        bx = self.param("bx", init.zeros_init(), (nx,), self.param_dtype)
        bv = self.param("bv", init.zeros_init(), (nv,), self.param_dtype)
        by = self.param("by", init.zeros_init(), (ny,), self.param_dtype)
        
        # Set some other params to zero
        D12 = self.param("D12", init.zeros_init(), (nv, nu), self.param_dtype)
        D21 = self.param("D21", init.zeros_init(), (ny, nv), self.param_dtype)
        
        # Irrelevant params for contracting REN
        X3 = self.param("X3", init.zeros_init(), (0,), self.param_dtype)
        Y3 = self.param("Y3", init.zeros_init(), (0,), self.param_dtype)
        Z3 = self.param("Z3", init.zeros_init(), (0,), self.param_dtype)
        
        # Build up the X matrix based on the initial state-space model
        P = solve_discrete_lyapunov_direct(A.T, jnp.identity(nx))
        Lambda = jnp.identity(nv)
        PA = P @ A
        
        H = jnp.block([
            [P, jnp.zeros((nx, nv)), PA.T],
            [jnp.zeros((nv, nx)), 2*Lambda, jnp.zeros((nv, nx))],
            [PA, jnp.zeros((nx, nv)), P]
        ])
        H = H + self.eps * jnp.identity(2 * nx + nv) # TODO: Add Wishart?
        
        X = jnp.linalg.cholesky(H, upper=True)
        X = self.param("X", init.constant(X), (2*nx + nv, 2*nx + nv), self.param_dtype)
        
        # Other implicit params
        X_norm = l2_norm(X, eps=self.eps)
        p = self.param("polar", init.constant(X_norm), (1,), self.param_dtype)        
        Y1 = self.param("Y1", identity_init(), (nx, nx), self.param_dtype)
        
        # Set fixed arrays directly from state matrices
        B2_imp = P @ B
        B2 = self.param("B2", init.constant(B2_imp), (nx, nu), self.param_dtype)
        C2 = self.param("C2", init.constant(C), (ny, nx), self.param_dtype)
        D22 = self.param("D22", init.constant(D), (ny, nu), self.param_dtype)
        
        # Set up the direct parameter struct
        self.direct = DirectRENParams(p, X, B2, D12, Y1, C2, D21, 
                                      D22, X3, Y3, Z3, bx, bv, by)
        
    def _direct_to_explicit(self, ps: DirectRENParams) -> ExplicitRENParams:
        H = self._x_to_h(ps.X, ps.p)
        explicit = self._hmatrix_to_explicit(ps, H, ps.D22)
        return explicit
    
    
class LipschitzREN(RENBase):
    """Construct a Lipschitz-bounded REN.
    
    Attributes::
        gamma: upper bound on the Lipschitz constant (default 1.0).
    
    Example usage::

        >>> import jax, jax.numpy as jnp
        >>> from robustnn import ren
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, nv, ny = 1, 2, 4, 1
        >>> model = ren.LipschitzREN(nu, nx, nv, ny, gamma=10.0)
        
        >>> batches = 5
        >>> states = model.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = model.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 1), 'X': (8, 8), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bv': (1, 4), 'bx': (1, 2), 'by': (1, 1), 'polar': (1,)}}
    
    See docs for `RENBase` for full list of arguments.
    """
    gamma: jnp.float32 = 1.0 # type: ignore
    
    def _error_checking(self):
        if self.d22_free:
            raise ValueError("Set `d22_free=False` for Lipschitz RENs.")
    
    def _direct_to_explicit(self, ps: DirectRENParams) -> ExplicitRENParams:
        nu = self.input_size
        nx = self.state_size
        ny = self.output_size
        Iu = jnp.identity(nu, self.param_dtype)
        Iy = jnp.identity(ny, self.param_dtype)
        
        
        # Implicit params
        B2_imp = ps.B2
        D12_imp = ps.D12
        
        # Construct D22 (Eqns 31-33 of Revay et al. (2023))
        if self.d22_zero:
            D22 = ps.D22
        else:
            M = ps.X3.T @ ps.X3 + ps.Y3 - ps.Y3.T + ps.Z3.T @ ps.Z3 + self.eps*Iy
            if ny >= nu:
                N = jnp.vstack((jnp.linalg.solve((Iy + M).T, (Iy - M).T).T,
                                jnp.linalg.solve((Iy + M).T, -2*ps.Z3.T).T))
            else:
                N = jnp.hstack((jnp.linalg.solve((Iy + M), (Iy - M)),
                                jnp.linalg.solve((Iy + M), -2*ps.Z3.T)))
            D22 = self.gamma * N
        
        # Construct H (Eqn. 28 of Revay et al. (2023))
        C2_imp = -D22.T @ ps.C2 / self.gamma
        D21_imp = -(D22.T @ ps.D21) / self.gamma - D12_imp.T
        
        R = self.gamma * (-D22.T @ D22 / (self.gamma**2) + Iu)
        mul_Q = jnp.hstack((ps.C2, ps.D21, jnp.zeros((ny, nx), self.param_dtype)))
        mul_R = jnp.hstack((C2_imp, D21_imp, B2_imp.T))
        Gamma_Q = mul_Q.T @ mul_Q / (-self.gamma)
        Gamma_R = mul_R.T @ jnp.linalg.solve(R, mul_R)
        
        H = self._x_to_h(ps.X, ps.p) + Gamma_R - Gamma_Q
        explicit = self._hmatrix_to_explicit(ps, H, D22)
        return explicit


class GeneralREN(RENBase):
    """Construct a REN satisfying an incremental IQC defined by Q, S, R.
    
    Example usage::

        >>> import jax, jax.numpy as jnp
        >>> from robustnn import ren
        
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
        >>> model = ren.GeneralREN(nu, nx, nv, ny, Q=Q, S=S, R=R)
        >>> model.check_valid_qsr()
        
        >>> batches = 5
        >>> states = model.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = model.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 1), 'X': (8, 8), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bv': (1, 4), 'bx': (1, 2), 'by': (1, 1), 'polar': (1,)}}
        
    Attributes::
        Q: IQC output weight.
        S: IQC cross input/output weight.
        R: IQC input weight.
    
    The IQC matrices have the following conditions for a REN with input
    size `nu` and output size `ny`:
        - `Q.shape` must be `(ny, ny)`.
        - `S.shape` must be `(nu, ny)`.
        - `R.shape` must be `(nu, nu)`.
        - `Q` must be negative definite.
        - `R - S @ (inv(Q) @ S.T)` must be positive definite.
    We expect users to JIT calls to the `.init()` and `.apply()` methods for a
    REN, so we leave error checking as a separate API call. Use 
    `model.check_valid_qsr(*model.qsr)` to check for appropriate (Q, S, R) matrices.
    """
    Q: Array = None
    S: Array = None
    R: Array = None
    
    def _error_checking(self):
        if self.d22_free:
            raise ValueError("Set `d22_free=False` for general QSR RENs")
        if (not self.d22_zero) and self.init_output_zero:
            raise ValueError("Cannot have zero output on init without setting `d22_zero=True`.")
        
    def _direct_to_explicit(self, ps: DirectRENParams) -> ExplicitRENParams:
        nu = self.input_size
        nx = self.state_size
        ny = self.output_size
        Q, S, R = self._adjust_iqc_params()
        
        # Compute useful decompositions
        R_temp = R - S @ jnp.linalg.solve(Q, S.T)
        LQ = jnp.linalg.cholesky(-Q, upper=True)
        LR = jnp.linalg.cholesky(R_temp, upper=True)
        
        # Implicit params
        B2_imp = ps.B2
        D12_imp = ps.D12
        
        # Construct D22 (Eqns 31-33 of Revay et al. (2023))
        if self.d22_zero:
            D22 = ps.D22
        else:
            I = jnp.identity(ny, self.param_dtype)
            M = ps.X3.T @ ps.X3 + ps.Y3 - ps.Y3.T + ps.Z3.T @ ps.Z3 + self.eps*I
            if ny >= nu:
                N = jnp.vstack((jnp.linalg.solve((I + M).T, (I - M).T).T,
                                jnp.linalg.solve((I + M).T, -2*ps.Z3.T).T))
            else:
                N = jnp.hstack((jnp.linalg.solve((I + M), (I - M)),
                                jnp.linalg.solve((I + M), -2*ps.Z3.T)))
            
            D22 = jnp.linalg.solve(-Q, S.T) + jnp.linalg.solve(LQ, N) @ LR
        
        # Construct H (Eqn. 28 of Revay et al. (2023))
        C2_imp = (D22.T @ Q + S) @ ps.C2
        D21_imp = (D22.T @ Q + S) @ ps.D21 - D12_imp.T
        
        R1 = R + S @ D22 + D22.T @ S.T + D22.T @ Q @ D22
        mul_Q = jnp.hstack((ps.C2, ps.D21, jnp.zeros((ny, nx), self.param_dtype)))
        mul_R = jnp.hstack((C2_imp, D21_imp, B2_imp.T))
        Gamma_Q = mul_Q.T @ Q @ mul_Q
        Gamma_R = mul_R.T @ jnp.linalg.solve(R1, mul_R)
        
        H = self._x_to_h(ps.X, ps.p) + Gamma_R - Gamma_Q
        explicit = self._hmatrix_to_explicit(ps, H, D22)
        return explicit

    def check_valid_qsr(self):
        """Check that the (Q,S,R) matrices are valid.
        
        Example usage:
            >>> Q, S, R = ... # Define your matrices here.
            
            >>> nu, nx, nv, ny = 1, 3, 4, 2
            >>> ren = GeneralREN(nu, nx, nv, ny, Q=Q, S=S, R=R)
            >>> ren.check_valid_qsr()
            
        This function is NOT run automatically in the `setup()` routine
        to avoid issues with the JAX tracer.
        """
        nu = self.input_size
        ny = self.output_size
        Q, S, R = self._adjust_iqc_params()
        
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
        
    def _adjust_iqc_params(self):
        """Small delta to help numerical conditioning with cholesky decomposition."""
        Q = self.Q - self.eps * jnp.identity(self.Q.shape[0], self.param_dtype)
        R = self.R + self.eps * jnp.identity(self.R.shape[0], self.param_dtype)
        return Q, self.S, R


def _check_posdef(A: Array, eps=jnp.finfo(jnp.float32).eps):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        return False
    A = A + eps * jnp.identity(A.shape[0], A.dtype)
    if not jnp.allclose(A, A.T):
        return False
    if not jnp.all(jnp.linalg.eigh(A)[0] > 0.0):
        return False
    return True
