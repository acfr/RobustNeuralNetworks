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


def get_valid_init():
    return ["random", "long_memory"]
    

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
        

@dataclass
class DirectRENParams:
    """Data class to keep track of direct params for a REN.
    
    These are the free, trainable parameters for a REN.
    """
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
    """Data class to keep track of explicit params for a REN.
    
    These are the parameters used for evaluating a REN.
    """
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
        input_size: number of input features (nu).
        state_size: number of internal states (nx).
        features: number of (hidden) neurons (nv).
        output_size: number of output features (ny).
        activation: Activation function to use (default: relu).
        
        kernel_init: initializer for weights (default: lecun_normal()).
        recurrent_kernel_init: initializer for the REN `X` matrix (default: lecun_normal()).
        carry_init: initializer for the internal state vector (default: zeros_init()).
        x_bias_init: initializer for the state bias parameters (default: zeros_init()).
        v_bias_init: initializer for the feedback bias parameters (default: zeros_init()).
        y_bias_init: initializer for the output bias parameters (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        
        init_method: parameter initialisation method to choose from. Options are:
        
        - "random" (default): Random sampling with `recurrent_kernel_init`.
        - "long_memory": Initialise such that `A = I` (approx.) in explicit model.
            Good for long-memory dynamics on initialisation.
        
        init_output_zero: initialize the network so its output is zero (default: False).
        identity_output: enforce that output layer is ``y_t = x_t``. (default: False).
            
        do_polar_param: Use the polar parameterization for the H matrix (default: True).
        d22_zero: Fix `D22 = 0` to remove any feedthrough in the REN (default: False).
        abar: upper bound on the contraction rate. Requires `0 <= abar <= 1` (default: 1).
        eps: regularising parameter for positive-definite matrices (default: machine 
            precision for `jnp.float32`).
    """
    input_size: int     # nu
    state_size: int     # nx
    features: int       # nv
    output_size: int    # ny
    activation: ActivationFn = nn.relu
    
    kernel_init: Initializer = init.lecun_normal()
    recurrent_kernel_init: Initializer = init.lecun_normal()
    carry_init: Initializer = init.zeros_init()
    x_bias_init: Initializer = init.zeros_init()
    v_bias_init: Initializer = init.zeros_init()
    y_bias_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    
    init_method: str = "random"
    init_output_zero: bool = False
    identity_output: bool = False
    
    do_polar_param: bool = True
    d22_zero: bool = False
    abar: jnp.float32 = 1 # type: ignore
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps # type: ignore
    
    def setup(self):
        """Initialise the direct REN params."""

        # Error checking
        self._error_check_output_layer()
        self._error_checking()
        if self.init_method not in get_valid_init():
            raise ValueError("Undefined init method '{}'".format(self.init_method))

        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        dtype = self.param_dtype
        
        # Define direct params for REN
        B2 = self.param("B2", self.kernel_init, (nx, nu), dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), dtype)
        
        bx = self.param("bx", self.x_bias_init, (nx,), dtype)
        bv = self.param("bv", self.v_bias_init, (nv,), dtype)
        
        # Special construction for X matrix
        if self.init_method == "random":
            x_init = self.recurrent_kernel_init
            Y1 = self.param("Y1", self.kernel_init, (nx, nx), dtype)
            
        elif self.init_method == "long_memory":
            x_init = self._x_long_memory_init(B2, D12)
            Y1 = self.param("Y1", init.constant(jnp.identity(nx)), (nx, nx), dtype)
        
        X = self.param("X", x_init, (2*nx + nv, 2*nx + nv), dtype)
        p = self.param("p", init.constant(l2_norm(X, eps=self.eps)), (1,), dtype)
        
        # Output layer params
        if self.init_output_zero:
            out_kernel_init = init.zeros_init()
            out_bias_init = init.zeros_init()
        else:
            out_kernel_init = self.kernel_init
            out_bias_init = self.y_bias_init
        
        if self.identity_output:
            C2 = jnp.identity(nx)
            D21 = jnp.zeros((ny, nv), dtype)
            by = jnp.zeros((ny,), dtype)
        else:
            by = self.param("by", out_bias_init, (ny,), dtype)
            C2 = self.param("C2", out_kernel_init, (ny, nx), dtype)
            D21 = self.param("D21", out_kernel_init, (ny, nv), dtype)
            D22 = self.param("D22", init.zeros_init(), (ny, nu), dtype)
                
        if self.identity_output or self.d22_zero:
            D22 = jnp.zeros((ny, nu), dtype)
            
        # These parameters are used to construct D22 instead of the above for most RENs.
        # Could tidy up the code a little here by not initialising D22 at all.
        # By default they initialise D22 = 0
        d = min(nu, ny)
        X3 = self.param("X3", identity_init(), (d, d), dtype)
        Y3 = self.param("Y3", init.zeros_init(), (d, d), dtype)
        Z3 = self.param("Z3", init.zeros_init(), (abs(ny - nu), d), dtype)
            
        # Set up the direct parameter struct
        self.direct = DirectRENParams(p, X, B2, D12, Y1, C2, D21, 
                                      D22, X3, Y3, Z3, bx, bv, by)
        
    def _x_long_memory_init(self, B2: Array, D12: Array):
        """Initialise the X matrix so E, F, P (and therefore A) are I.

        Args:
            B2 (Array): B2 matrix (used in init).
            D12 (Array): D12 matrix (used in init).
            
        Returns:
            function: initialiser function with signature 
                `init_func(key, shape, dtype) -> Array`
        """
        def init_func(key, shape, dtype) -> Array:
            dtype = self.param_dtype
            key, rng1, rng2 = jax.random.split(key, 3)
            
            nx = B2.shape[0]
            nv = D12.shape[0]
            
            eigs = 0.05 * jax.random.uniform(rng1, (nx,))
            
            E = jnp.identity(nx, dtype)
            F = jnp.identity(nx, dtype) - jnp.diag(eigs)
            P = jnp.identity(nx, dtype)
            
            B1 = jnp.zeros((nx, nv), dtype)
            C1 = jnp.zeros((nv, nx), dtype)
            D11 = self.kernel_init(rng2, (nv, nv), dtype)
            
            # Need eigvals of Lambda large enough so that H22 is pos def
            eigs, _ = jnp.linalg.eigh(D11 + D11.T)
            Lambda = (jnp.max(eigs) / 2 + 1e-4) * jnp.identity(nv, dtype)
            H22 = 2*Lambda - D11 - D11.T
            
            H = jnp.block([
                [(E + E.T - P), -C1.T, F.T],
                [-C1, H22, B1.T],
                [F, B1, P]
            ]) + self.eps * jnp.identity(shape[0])
            
            X = jnp.linalg.cholesky(H, upper=True)
            return X
        
        return init_func
        
    def __call__(self, state: Array, inputs: Array) -> Tuple[Array, Array]:
        """Call a REN model

        Args:
            state (Array): internal model state.
            inputs (Array): model inputs.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(state, inputs, explicit)
    
    def _explicit_call(
        self, x: Array, u: Array, e: ExplicitRENParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for a REN.

        Args:
            x (Array): internal model state.
            u (Array): model inputs.
            e (ExplicitRENParams): explicit params.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        b = x @ e.C1.T + u @ e.D12.T + e.bv
        w = tril_equlibrium_layer(self.activation, e.D11, b)
        x1 = x @ e.A.T + w @ e.B1.T + u @ e.B2.T + e.bx
        y = x @ e.C2.T + w @ e.D21.T + u @ e.D22.T + e.by
        return x1, y
    
    def _simulate_sequence(self, x0, u) -> Tuple[Array, Array]:
        """Simulate a REN over a sequence of inputs.

        Args:
            params (dict): Flax model parameters dictionary.
            x0: array of initial states, shape is (batches, ...).
            u: array of inputs as a sequence, shape is (time, batches, ...).
            
        Returns:
            Tuple[Array, Array]: (final_state, outputs in (time, batches, ...)).
        """
        explicit = self._direct_to_explicit()
        def rollout(carry, ut):
            xt, = carry
            xt1, yt = self._explicit_call(xt, ut, explicit)
            return (xt1,), yt
        (x1, ), y = jax.lax.scan(rollout, (x0,), u)
        return x1, y
    
    @nn.nowrap
    def initialize_carry(
        self, rng: jax.Array, input_shape: Tuple[int, ...]
    ) -> Array:
        """Initialise the REN state (carry).

        Args:
            rng (jax.Array): random seed for carry initialisation.
            input_shape (Tuple[int, ...]): Shape of model input array.

        Returns:
            Array: initial model state.
        """
        batch_dims = input_shape[:-1]
        rng, _ = jax.random.split(rng)
        mem_shape = batch_dims + (self.state_size,)
        return self.carry_init(rng, mem_shape, self.param_dtype)
    
    def _hmatrix_to_explicit(
        self, ps: DirectRENParams, H: Array, D22: Array
    ) -> ExplicitRENParams:
        """Convert REN H matrix to explict model given direct params.

        Args:
            ps (DirectRENParams): direct REN params.
            H (Array): REN H matrix used in parameterisation (see Eqns. 19, 29).
            D22 (Array): The D22 matrix to be used. Allows for special construction.

        Returns:
            ExplicitRENParams: explicit REN model.
        """
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
        
        # Biases can go unchanged
        bx_e = ps.bx
        bv_e = ps.bv
        by_e = ps.by
        
        # Remaining explicit params are biases/in the output layer (unchanged)
        explicit = ExplicitRENParams(A_e, B1_e, B2_e, C1_e, ps.C2, D11_e, 
                                     D12_e, ps.D21, D22, bx_e, bv_e, by_e)
        return explicit
    
    def _x_to_h_contracting(self, X: Array, p: Array) -> Array:
        """Convert REN X matrix to part of H matrix used in the contraction
        setup (using polar parameterization if required).

        Args:
            X (Array): REN X matrix.
            p (Array): polar parameter.

        Returns:
            Array: REN H matrix.
        """
        H = X.T @ X
        if self.do_polar_param:
            H = p**2 * H / (l2_norm(X)**2)
        return H + self.eps * jnp.identity(jnp.shape(X)[0])
    
    
    ############### Specify these for each REN parameterisation ###############
    
    def _error_checking(self):
        """Check conditions for REN."""
        pass
        
    def _direct_to_explicit(self) -> ExplicitRENParams:
        """
        Convert direct paremeterization of a REN to explicit form
        for evaluation. This depends on the specific REN parameterization.
        """
        raise NotImplementedError(
            "RENBase models should not be called. " +
            "Choose a REN parameterization instead (eg: `ContractingREN`)."
        )
    
    
    #################### Error checking ####################
    
    def _error_check_output_layer(self):
        """Error checking for options on the output layer."""
        
        if self.init_output_zero and self.identity_output:
            raise ValueError("Cannot have zero output if identity output y_t = x_t is requested.")
        
        if self.identity_output:
            if self.state_size != self.output_size:
                raise ValueError(
                    "When output layer is identity map, need state_size == output_size."
                )


    #################### Convenient Wrappers ####################

    def explicit_call(
        self, params:dict, x: Array, u: Array, e: ExplicitRENParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for a REN.

        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): internal model state.
            u (Array): model inputs.
            e (ExplicitRENParams): explicit params.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        # Don't need to use .apply() for REN, it doesn't need to access
        # anything that's defined in the setup() method.
        # return self.apply(params, x, u, e, method="_explicit_call")
        return self._explicit_call(x, u, e)
    
    def direct_to_explicit(self, params: dict) -> ExplicitRENParams:
        """Convert from direct to explicit REN params.

        Args:
            params (dict): Flax model parameters dictionary.

        Returns:
            ExplicitRENParams: explicit params for REN.
        """
        return self.apply(params, method="_direct_to_explicit")
    
    def simulate_sequence(self, params: dict, x0, u) -> Tuple[Array, Array]:
        """Simulate a REN over a sequence of inputs.

        Args:
            params (dict): Flax model parameters dictionary.
            x0: array of initial states, shape is (batches, ...).
            u: array of inputs as a sequence, shape is (time, batches, ...).
            
        Returns:
            Tuple[Array, Array]: (final_state, outputs in (time, batches, ...)).
            
        Note:
            - Use this if you would otherwise do `model.apply()` in a loop.
            - The direct -> explicit map is only called once, at the start
            of the sequence. This avoids unnecessary calls to the parameter
            mapping and should speed up your code :)
        """
        return self.apply(params, x0, u, method="_simulate_sequence")
