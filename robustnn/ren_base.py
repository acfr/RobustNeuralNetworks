import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array

from robustnn.utils import l2_norm, identity_init
from robustnn.utils import ActivationFn, Initializer


@dataclass
class DirectRENParams:
    """Class to keep track of explicit params for a REN."""
    p: Array
    X: Array
    B2: Array
    D12: Array
    Y1: Array
    C2: Array
    D21: Array
    D22: Array
    X3: Array
    Y3: Array
    Z3: Array
    bx: Array
    bv: Array
    by: Array


@dataclass
class ExplicitRENParams:
    """Class to keep track of explicit params for a REN."""
    A: Array
    B1: Array
    B2: Array
    C1: Array
    C2: Array
    D11: Array
    D12: Array
    D21: Array
    D22: Array
    bx: Array
    bv: Array
    by: Array


class RENBase(nn.Module):
    """
    Base class for Recurrent Equilibrium Networks (RENs).
    
    The attributes are labelled similarly to `nn.LSTM` for
    convenience, but this deviates from the REN literature.
    Explanations below.
        
    Attributes::
        input_size: the number of input features (nu).
        state_size: the number of internal states (nx).
        features: the number of (hidden) neurons (nv).
        output_size: the number of output features (ny).
        activation: Activation function to use (default: relu).
        kernel_init: initializer for weights (default: lecun_normal()).
        recurrent_kernel_init: initializer for the REN `X` matrix (default: lecun_normal()).
        bias_init: initializer for the bias parameters (default: zeros_init()).
        carry_init: initializer for the internal state vector (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        init_method: parameter initialisation method to choose from. Options are:
            - `"random"` (default): Random sampling with `recurrent_kernel_init`.
            - `"cholesky"`: Compute `X` with cholesky factorisation of `H`, sets `E,F,P = 
                            I`. Good for slow/long memory dynamic models.
        init_output_zero: initialize the network so its output is zero (default: False).
        d22_free: Specify whether to train `D22` as a free parameter (`True`), or construct
                  it separately from `X3, Y3, Z3` (`false`). Typically `True` only for a 
                  contracting REN (default: False).
        d22_zero: Fix `D22 = 0` to remove any feedthrough in the REN (default: False).
        eps: Regularising parameter for positive-definite matrices (default: machine 
             precision for `jnp.float32`).
        abar: upper bound on the contraction rate. Requires
              `0 <= abar <= 1` (default: 1).
        
    NOTE: Initializing the `X` matrix for a REN with `orthogonal()` is likely to make
          the initial REN dynamics slow, with long memory. For faster initial dynamics,
          use `glorot_normal()` instead.
    """
    input_size: int
    state_size: int
    features: int
    output_size: int
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = init.lecun_normal()
    recurrent_kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()
    carry_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    init_method: str = "random"
    init_output_zero: bool = False
    d22_free: bool = False
    d22_zero: bool = False
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps # type: ignore
    abar: jnp.float32 = 1 # type: ignore
    
    def setup(self):
        """
        Initialise the direct parameters for a REN and perform error checking.
        """
        self._error_checking()
        self._init_params()

    def __call__(self, state: Array, inputs: Array) -> Tuple[Array, Array]:
        """
        Call a REN.
        
        This implementation treats the REN as a dynamical system to be evaluated
        at a single time. The syntax is `state, out = ren(state, in)`.
        """
        
        # Direct parameterisation mapping
        explicit = self._direct_to_explicit(self.direct)
        
        # Call the explicit REN form and return
        state, out = self.explicit_call(state, inputs, explicit)
        return state, out
    
    def explicit_call(
        self, x: Array, u: Array, e: ExplicitRENParams
    ) -> Tuple[Array, Array]:
        """
        Evaluate a REN given its explicit parameterization.
        """
        b = x @ e.C1.T + u @ e.D12.T + e.bv
        w = tril_equlibrium_layer(self.activation, e.D11, b)
        x1 = x @ e.A.T + w @ e.B1.T + u @ e.B2.T + e.bx
        y = x @ e.C2.T + w @ e.D21.T + u @ e.D22.T + e.by
        return x1, y
    
    def params_to_explicit(self, ps: dict):
        """
        Convert a parameter dictionary returned by the `.init()` method
        in Flax into an `ExplicitRENParams` instance.
        
        The `ps` dictionary must be of the form (with possibly different sizes):
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 1), 'X': (8, 8), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bv': (1, 4), 'bx': (1, 2), 'by': (1, 1), 'polar': (1,)}}
        """
        direct = DirectRENParams(
            p = ps["params"]["polar"],
            X = ps["params"]["X"],
            B2 = ps["params"]["B2"],
            D12 = ps["params"]["D12"],
            Y1 = ps["params"]["Y1"],
            C2 = ps["params"]["C2"],
            D21 = ps["params"]["D21"],
            D22 = ps["params"]["D22"],
            X3 = ps["params"]["X3"],
            Y3 = ps["params"]["Y3"],
            Z3 = ps["params"]["Z3"],
            bx = ps["params"]["bx"],
            bv = ps["params"]["bv"],
            by = ps["params"]["by"]
        )
        return self._direct_to_explicit(direct)
    
    @nn.nowrap
    def initialize_carry(
        self, rng: jax.Array, input_shape: Tuple[int, ...]
    ) -> Array:
        """Initialize the REN state (carry).
        
        Args:
        rng: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the network.
        
        Returns:
        An initialized state (carry) vector for the REN network.
        """
        batch_dims = input_shape[:-1]
        rng, _ = jax.random.split(rng)
        mem_shape = batch_dims + (self.state_size,)
        return self.carry_init(rng, mem_shape, self.param_dtype)
    
    def _init_params(self):
        """Initialise all direct params for a REN and store."""
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        
        # Define direct params for REN
        B2 = self.param("B2", self.kernel_init, (nx, nu), self.param_dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), self.param_dtype)
        
        if self.init_method == "random":
            x_init = self.recurrent_kernel_init
        elif self.init_method == "cholesky":
            x_init = self._x_cholesky_init(B2, D12, self.eps)
        else:
            raise ValueError("Undefined init method '{}'".format(self.init_method))
            
        X = self.param("X", x_init, (2 * nx + nv, 2 * nx + nv), self.param_dtype)
        
        p = self.param("polar", init.constant(l2_norm(X, eps=self.eps)), (1,), self.param_dtype)        
        Y1 = self.param("Y1", self.kernel_init, (nx, nx), self.param_dtype)
        
        bx = self.param("bx", self.bias_init, (nx,), self.param_dtype)
        bv = self.param("bv", self.bias_init, (nv,), self.param_dtype)
        
        # Output layer params
        if self.init_output_zero:
            out_kernel_init = init.zeros_init()
            out_bias_init = init.zeros_init()
        else:
            out_kernel_init = self.kernel_init
            out_bias_init = self.bias_init
            
        by = self.param("by", out_bias_init, (ny,), self.param_dtype)
        C2 = self.param("C2", out_kernel_init, (ny, nx), self.param_dtype)
        D21 = self.param("D21", out_kernel_init, (ny, nv), self.param_dtype)
        
        # The rest is for the feedthrough term D22
        D22 = self.param("D22", init.zeros_init(), (ny, nu), self.param_dtype)
        if self.d22_zero:
            _rng = jax.random.PRNGKey(0)
            D22 = init.zeros(_rng, (ny, nu), self.param_dtype)
        
        d = min(nu, ny)
        X3 = self.param("X3", identity_init(), (d, d), self.param_dtype)
        Y3 = self.param("Y3", init.zeros_init(), (d, d), self.param_dtype)
        Z3 = self.param("Z3", init.zeros_init(), (abs(ny - nu), d), self.param_dtype)
            
        # Set up the direct parameter struct
        self.direct = DirectRENParams(p, X, B2, D12, Y1, C2, D21, 
                                      D22, X3, Y3, Z3, bx, bv, by)
        
    def _x_cholesky_init(self, B2, D12, eps):
        """Initialise the X matrix so E, F, P are identity."""
        
        def init_func(key, shape, dtype) -> Array:
            
            key, rng = jax.random.split(key, 2)
            
            nx = B2.shape[0]
            nv = D12.shape[0]
            
            E = jnp.identity(nx, dtype)
            F = jnp.identity(nx, dtype)
            P = jnp.identity(nx, dtype)
            
            B1 = jnp.zeros((nx, nv), dtype)
            C1 = jnp.zeros((nv, nx), dtype)
            D11 = self.kernel_init(rng, (nv, nv), dtype)
            
            eigs, _ = jnp.linalg.eigh(D11 + D11.T)
            Lambda = (jnp.max(eigs) / 2 + 1e-4) * jnp.identity(nv, dtype)
            H22 = 2*Lambda - D11 - D11.T
            
            H = jnp.block([
                [(E + E.T - P), -C1.T, F.T],
                [-C1, H22, B1.T],
                [F, B1, P]
            ]) + eps * jnp.identity(shape[0])
            
            X = jnp.linalg.cholesky(H, upper=True)
            return X
        
        return init_func
    
    def _error_checking(self):
        """Check conditions for REN."""
        raise NotImplementedError("Each REN parameterisation should have its own version of this function.")
        
    def _direct_to_explicit(self, direct: DirectRENParams) -> ExplicitRENParams:
        """
        Convert direct paremeterization of a REN to explicit form
        for evaluation. This depends on the specific REN parameterization.
        """
        raise NotImplementedError("RENBase models should not be called. Choose a REN parameterization instead (eg: `ContractingREN`).")
    
    def _x_to_h(self, X: Array, p: Array) -> Array:
        """Convert REN X matrix to H matrix using polar parameterization."""
        H = p**2 * (X.T @ X) / (l2_norm(X)**2) + self.eps * jnp.identity(jnp.shape(X)[0])
        return H
    
    def _hmatrix_to_explicit(
        self, ps: DirectRENParams, H: Array, D22: Array
    ) -> ExplicitRENParams:
        """Convert REN H matrix to explict form given direct params."""
        
        nx = self.state_size
        nv = self.features
        
        # Extract sections of the H matrix
        H11 = H[:nx, :nx]
        H22 = H[nx:(nx + nv), nx:(nx + nv)]
        H33 = H[(nx + nv):(2*nx + nv), (nx + nv):(2*nx + nv)]
        H21 = H[nx:(nx + nv), :nx]
        H31 = H[(nx + nv):(2*nx + nv), :nx]
        H32 = H[(nx + nv):(2*nx + nv), nx:(nx + nv)]
                
        # Construct implicit model parameters
        P_imp = H33
        F = H31
        E = (H11 + P_imp / (self.abar**2) + ps.Y1 - ps.Y1.T) / 2
        
        # Equilibrium network params (imp for "implicit")
        B1_imp = H32
        C1_imp = -H21
        Lambda_inv = 2 / jnp.diag(H22)
        D11_imp = -jnp.tril(H22, k=-1)
        
        # Construct the explicit model (e for "explicit")
        A_e = jnp.linalg.solve(E, F)
        B1_e = jnp.linalg.solve(E, B1_imp)
        B2_e = jnp.linalg.solve(E, ps.B2)
        
        # Equilibrium layer matrices
        C1_e = (Lambda_inv * C1_imp.T).T
        D11_e = (Lambda_inv * D11_imp.T).T
        D12_e = (Lambda_inv * ps.D12.T).T
        
        # Remaining explicit params are biases/in the output layer (unchanged)
        explicit = ExplicitRENParams(A_e, B1_e, B2_e, C1_e, ps.C2, D11_e, 
                                     D12_e, ps.D21, D22, ps.bx, ps.bv, ps.by)
        return explicit
    

@partial(jax.jit, static_argnums=(0,))
def tril_equlibrium_layer(activation, D11, b):
    """
    Solve `w = activation(D11 @ w + b)` for lower-triangular D11.
    
    Activation must be monotone with slope restricted to `[0,1]`.
    """
    w_eq = jnp.zeros_like(b)
    D11_T = D11.T
    for i in range(D11.shape[0]):
        Di_T = D11_T[:i, i]
        wi = w_eq[..., :i]
        bi = b[..., i]
        Di_wi = wi @ Di_T
        w_eq = w_eq.at[..., i].set(activation(Di_wi + bi))
    return w_eq
