import jax
import jax.numpy as jnp

from dataclasses import dataclass
from flax import linen as nn
from flax.linen import initializers as init
from flax.typing import Dtype
from typing import Union, Callable, Any, Tuple

from robustnn.ren_jax.utils import tril_equlibrium_layer


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]
Array = Union[jax.Array, Any]


# (TODO) Avoid code repetition, this is the same as for LBDN
def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.maximum(jnp.sum(x**2, **kwargs), eps))


def identity_init():
    """Initialize a weight as the identity matrix.
    
    Assumes that shape is a tuple (n,n), only uses first element.
    """
    def init(key, shape, dtype) -> Array:
        return jnp.identity(shape[0], dtype)
    return init


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
        
    Attributes:
        input_size: the number of input features (nu).
        state_size: the number of internal states (nx).
        features: the number of (hidden) neurons (nv).
        output_size: the number of output features (ny).
        activation: Activation function to use (default: relu).
        kernel_init: initializer for weights (default: glorot_normal()).
        recurrent_kernel_init: initializer for the REN `X` matrix (default: orthogonal()).
        bias_init: initializer for the bias parameters (default: zeros_init()).
        carry_init: initializer for the internal state vector (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).
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
    kernel_init: Initializer = init.glorot_normal()
    recurrent_kernel_init: Initializer = init.orthogonal()
    bias_init: Initializer = init.zeros_init()
    carry_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    d22_free: bool = False
    d22_zero: bool = False
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps
    abar: jnp.float32 = 1
    
    @nn.compact
    def __call__(self, state: Array, inputs: Array) -> Tuple[Array, Array]:
        """
        Call an REN.
        
        This implementation treats the REN as a dynamical system
        to be evaluated at a single time. The syntax is
            state, out = ren(state, in)
        where `state` is an array.
        """
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        
        # Define REN params
        B2 = self.param("B2", self.kernel_init, (nx, nu), self.param_dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), self.param_dtype)
        X = self.param("X", self.recurrent_kernel_init, 
                       (2 * nx + nv, 2 * nx + nv), self.param_dtype)
        p = self.param("polar", init.constant(l2_norm(X, eps=self.eps)),
                       (1,), self.param_dtype)
        
        Y1 = self.param("Y1", self.kernel_init, (nx, nx), self.param_dtype)
        C2 = self.param("C2", self.kernel_init, (ny, nx), self.param_dtype)
        D21 = self.param("D21", self.kernel_init, (ny, nv), self.param_dtype)
        D22 = self.param("D22", init.zeros_init(), (ny, nu), self.param_dtype)
        if self.d22_zero:
            _rng = jax.random.PRNGKey(0)
            D22 = init.zeros(_rng, (ny, nu), self.param_dtype)
        
        if not self.d22_free and not self.d22_zero:
            d = min(nu, ny)
            X3 = self.param("X3", identity_init(), (d, d), self.param_dtype)
            Y3 = self.param("Y3", init.zeros_init(), (d, d), self.param_dtype)
            Z3 = self.param("Z3", init.zeros_init(), (abs(ny - nu), d), 
                            self.param_dtype)
        else:
            X3 = None
            Y3 = None
            Z3 = None
            
        bx = self.param("bx", self.bias_init, (1, nx), self.param_dtype)
        bv = self.param("bv", self.bias_init, (1, nv), self.param_dtype)
        by = self.param("by", self.bias_init, (1, ny), self.param_dtype)

        # Direct parameterisation mapping
        direct = DirectRENParams(p, X, B2, D12, Y1, C2, D21, 
                                 D22, X3, Y3, Z3, bx, bv, by)
        explicit = self.direct_to_explicit(direct)
        
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
    
    def direct_to_explicit(self, direct: DirectRENParams) -> ExplicitRENParams:
        """
        Convert direct paremeterization of a REN to explicit form
        for evaluation. This depends on the specific REN parameterization.
        """
        raise NotImplementedError("RENBase models should not be called. Choose a REN parameterization instead (eg: `ContractingREN`).")
    
    def x_to_h(self, X: Array, p: Array) -> Array:
        """Convert REN X matrix to H matrix using polar parameterization."""
        H = p**2 * (X.T @ X) / (l2_norm(X)**2) + self.eps * jnp.identity(jnp.shape(X)[0])
        return H
    
    def hmatrix_to_explicit(
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