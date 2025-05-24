import jax
import jax.numpy as jnp

from typing import Tuple, Sequence

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array

from robustnn import lbdn
from robustnn.utils import l2_norm
from robustnn.utils import ActivationFn, Initializer


def get_valid_init():
    return ["random", "long_memory"]


@dataclass
class DirectR2DNParams:
    """Data class to keep track of direct params for R2DN.
    
    These are the free, trainable parameters for an R2DN,
    excluding those in the LBDN layer.
    """
    p: Array
    X: Array
    Y: Array
    B1: Array
    B2: Array
    C1: Array
    D12: Array
    C2: Array
    D21: Array
    D22: Array
    bx: Array
    bv: Array
    by: Array
    network_params: lbdn.DirectLBDNParams


@dataclass
class ExplicitR2DNParams:
    """Data class to keep track of explicit params for R2DN.
    
    These are the parameters used for evaluating an R2DN.
    """
    A: Array
    B1: Array
    B2: Array
    C1: Array
    C2: Array
    D12: Array
    D21: Array
    D22: Array
    bx: Array
    bv: Array
    by: Array
    network_params: lbdn.ExplicitLBDNParams
    

class ContractingR2DN(nn.Module):
    """Robust Recurrent Deep Network (R2DN).
    
    This structure replaces the equilibrium layer in a REN with a
    1-Lipschitz multi-layer perceptron.

    Attributes:
        input_size: number of input features (nu).
        state_size: number of internal states (nx).
        features: number of (hidden) neurons (nv).
        output_size: number of output features (ny).
        hidden: sequence of hidden layer sizes for 1-Lipschitz network.
        activation: Activation function to use (default: relu).
        
        kernel_init: initializer for weights (default: lecun_normal()).
        recurrent_kernel_init: initialiser for X matrix (default: lecun_normal()).
        carry_init: initializer for the internal state vector (default: zeros_init()).
        x_bias_init: initializer for the state bias parameters (default: zeros_init()).
        v_bias_init: initializer for the feedback bias parameters (default: zeros_init()).
        y_bias_init: initializer for the output bias parameters (default: zeros_init()).
        network_bias_init: initializer for the 1-Lipschitz network bias parameters 
            (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).

        init_method: parameter initialisation method to choose from. No other methods are 
            currently supported for the R2DN (TODO). Options are:
        
        - "random" (default): Random sampling with `recurrent_kernel_init`.
        - "long_memory": Initialise such that `A = I` (approx.) in explicit model.
            Good for long-memory dynamics on initialisation.
        
        init_output_zero: initialize the network so its output is zero (default: False).
        identity_output: enforce that output layer is ``y_t = x_t``. (default: False).
            
        do_polar_param: Use the polar parameterization for the H matrix (default: True).
        eps: regularising parameter for positive-definite matrices (default: machine 
            precision for `jnp.float32`).
        
    Example usage:

        >>> import jax, jax.numpy as jnp
        >>> from robustnn import r2dn
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, nv, ny = 1, 2, 4, 1
        >>> nh = (2, 4)
        >>> model = r2dn.ContractingR2DN(nu, nx, nv, ny, nh)
        
        >>> batches = 5
        >>> states = model.initialize_carry(key1, (batches, nu))
        >>> inputs = jnp.ones((batches, nu))
        
        >>> params = model.init(key2, states, inputs)
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 
        1), 'Xbar': (2, 12), 'Y1': (2, 2), 'Y2': (4, 4), 'Y3': (6, 4), 'bv': (4,), 'bx': 
        (2,), 'by': (1,), 'network': {'layers_0': {'XY': (6, 2), 'a': (1,), 'b': (2,), 'd': 
        (2,)}, 'layers_1': {'XY': (6, 4), 'a': (1,), 'b': (4,), 'd': (4,)}, 'layers_2': 
        {'XY': (8, 4), 'a': (1,), 'b': (4,), 'd': (4,)}, 'ln_gamma': (1,)}, 'p1': (1,), 
        'p2': (1,), 'p3': (1,)}}
    """
    
    input_size: int             # nu
    state_size: int             # nx
    features: int               # nv
    output_size: int            # ny
    hidden: Sequence[int]       # Hidden layer sizes in the LBDN
    activation: ActivationFn = nn.relu
    
    kernel_init: Initializer = init.lecun_normal()
    recurrent_kernel_init: Initializer = init.lecun_normal()
    carry_init: Initializer = init.zeros_init()
    x_bias_init: Initializer = init.zeros_init()
    v_bias_init: Initializer = init.zeros_init()
    y_bias_init: Initializer = init.zeros_init()
    network_bias_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    
    init_method: str = "random"
    init_output_zero: bool = False
    identity_output: bool = False
    
    do_polar_param: bool = True
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps # type: ignore
    _gamma: jnp.float32 = 1.0 # type: ignore
    
    def setup(self):
        """Initialise all direct params for an R2DN and store."""
        
        if self.init_method not in get_valid_init():
            raise ValueError("Undefined init method '{}'".format(self.init_method))
        
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        dtype = self.param_dtype
        
        # Initialise an LBDN for the nonlinear layer
        self._network_init()
        
        # Initialise free parameters        
        B2 = self.param("B2", self.kernel_init, (nx, nu), dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), dtype)
        bx = self.param("bx", self.x_bias_init, (nx,), dtype)
        bv = self.param("bv", self.v_bias_init, (nv,), dtype)
        
        # Long-horizon initialisation or not
        if self.init_method == "random":
            x_init = self.recurrent_kernel_init
            Y = self.param("Y", self.kernel_init, (nx, nx), dtype)
            B1 = self.param("B1", self.kernel_init, (nx, nv), dtype)
            C1 = self.param("C1", self.kernel_init, (nv, nx), dtype)
            
        elif self.init_method == "long_memory":
            x_init = self._x_long_memory_init()
            Y = self.param("Y", init.constant(jnp.identity(nx)), (nx, nx), dtype)
            B1 = self.param("B1", init.zeros_init(), (nx, nv), dtype)
            C1 = self.param("C1", init.zeros_init(), (nv, nx), dtype)
            
        X = self.param("X", x_init, (2*nx, 2*nx), dtype)
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
            D22 = jnp.zeros((ny, nu), dtype)
            by = jnp.zeros((ny,), dtype)
        else:
            by = self.param("by", out_bias_init, (ny,), dtype)
            C2 = self.param("C2", out_kernel_init, (ny, nx), dtype)
            D21 = self.param("D21", out_kernel_init, (ny, nv), dtype)
            D22 = self.param("D22", init.zeros_init(), (ny, nu), dtype)
            
        self.direct = DirectR2DNParams(
            p, X, Y, B1, B2, C1, D12, C2, D21, D22, bx, bv, by, self.network.direct
        )        
    
    def _x_long_memory_init(self):
        """Initialise the X matrix so A is close to the identity.
        
        Assumes B1, C1 = 0 and Y = E = I.
        """
        def init_func(key, shape, dtype) -> Array:
            nx = self.state_size
            dtype = self.param_dtype
            
            key, rng = jax.random.split(key)
            eigs = 0.05 * jax.random.uniform(rng, (nx,))
            
            E = jnp.identity(nx, dtype)
            A = jnp.identity(nx, dtype) - jnp.diag(eigs)
            P = jnp.identity(nx, dtype)
            
            H = jnp.block([
                [(E + E.T - P), A.T],
                [A, P]
            ])
            
            X = jnp.linalg.cholesky(H, upper=True)
            return X
        
        return init_func
    
    def _network_init(self):
        """Initialise the LBDN for the nonlinear layer"""
        self.network = lbdn.LBDN(
            input_size=self.features,
            hidden_sizes=self.hidden,
            output_size=self.features,
            gamma=self._gamma,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.network_bias_init,
            param_dtype=self.param_dtype
        )
        
    def __call__(self, state: Array, inputs: Array) -> Tuple[Array, Array]:
        """Call an R2DN model

        Args:
            state (Array): internal model state.
            inputs (Array): model inputs.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        
        explicit = self._direct_to_explicit()
        return self._explicit_call(state, inputs, explicit)
        
    def _explicit_call(
        self, x: Array, u: Array, e: ExplicitR2DNParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for an R2DN.

        Args:
            x (Array): internal model state.
            u (Array): model inputs.
            e (ExplicitR2DNParams): explicit params.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """

        # Equilibirum layer
        v = x @ e.C1.T + u @ e.D12.T + e.bv
        w = self.network._explicit_call(v, e.network_params)
        
        # State-space model
        x1 = x @ e.A.T + w @ e.B1.T + u @ e.B2.T + e.bx
        y = x @ e.C2.T + w @ e.D21.T + u @ e.D22.T + e.by
        return x1, y
    
    def _simulate_sequence(self, x0, u) -> Tuple[Array, Array]:
        """Simulate an R2DN over a sequence of inputs.

        Args:
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
        """Initialise the R2DN state (carry).

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
        
    def _direct_to_explicit(self) -> ExplicitR2DNParams:
        """Convert from direct to explicit R2DN params.

        Args:
            None

        Returns:
            ExplicitR2DNParams: explicit params for R2DN.
        """
        ps = self.direct
        nx = self.state_size
        
        H = self._x_to_h_contracting(ps.X, ps.p, ps.B1, ps.C1)
        H11 = H[:nx, :nx]
        H21 = H[nx:, :nx]
        H22 = H[nx:, nx:]
        
        E = (H11 + H22 + ps.Y - ps.Y.T) / 2
        A = jnp.linalg.solve(E, H21)
        B1 = jnp.linalg.solve(E, ps.B1)
        
        return ExplicitR2DNParams(
            A, B1, ps.B2, ps.C1, ps.C2, ps.D12, ps.D21, ps.D22, ps.bx, ps.bv, ps.by,
            network_params = self.network._direct_to_explicit()
        )
            
    def _x_to_h_contracting(self, X: Array, p: Array, B1: Array, C1: Array) -> Array:
        """Convert R2DN X matrix to part of H matrix used in the contraction
        setup (using polar parameterization if required).

        Args:
            X (Array): REN X matrix.
            p (Array): polar parameter.
            B1 (Array): REN B1 matrix from implicit model.
            C1 (Array): REN C1 matrix from explicit model.

        Returns:
            Array: REN H matrix.
        """
        nx = jnp.shape(B1)[0]
        nX = jnp.shape(X)[0]
        
        H = X.T @ X
        if self.do_polar_param:
            H = p**2 * H / (l2_norm(X)**2)
            
        H = H + jnp.block([
            [C1.T @ C1, jnp.zeros((nx, nx))],
            [jnp.zeros((nx, nx)), B1 @ B1.T],
        ]) + self.eps * jnp.identity(nX)
        
        return H 
    
        
    #################### Convenient Wrappers ####################

    def explicit_call(
        self, params:dict, x: Array, u: Array, e: ExplicitR2DNParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for an R2DN.

        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): internal model state.
            u (Array): model inputs.
            e (ExplicitR2DNParams): explicit params.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        return self.apply(params, x, u, e, method="_explicit_call")
    
    def simulate_sequence(self, params: dict, x0, u) -> Tuple[Array, Array]:
        """Simulate an R2DN over a sequence of inputs.

        Args:
            params (dict): Flax model parameters dictionary.
            x0: array of initial states, shape is (batches, ...).
            u: array of inputs as a sequence, shape is (time, batches, ...).
            
        Returns:
            Tuple[Array, Array]: (final_state, outputs in (time, batches, ...)).
        """
        return self.apply(params, x0, u, method="_simulate_sequence")
    
    def direct_to_explicit(self, params: dict) -> ExplicitR2DNParams:
        """Convert from direct to explicit R2DN params.

        Args:
            params (dict): Flax model parameters dictionary.

        Returns:
            ExplicitR2DNParams: explicit params for R2DN.
        """
        return self.apply(params, method="_direct_to_explicit")
