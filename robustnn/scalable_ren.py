import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp

from typing import Tuple, Sequence

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array

from robustnn import lbdn
from robustnn.utils import l2_norm, solve_discrete_lyapunov_direct
from robustnn.utils import ActivationFn, Initializer


def get_valid_init():
    return ["random", "long_memory", "random_explicit", "long_memory_explicit", 
            "external_explicit"]


@dataclass
class DirectSRENParams:
    """Data class to keep track of direct params for Scalable REN.
    
    These are the free, trainable parameters for a Scalable REN,
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
class ExplicitSRENParams:
    """Data class to keep track of explicit params for Scalable REN.
    
    These are the parameters used for evaluating a Scalable REN.
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
    

class ScalableREN(nn.Module):
    """Scalable version of Recurrent Equilirbium Network.
    
    This structure replaces the equilibrium layer in the REN with a
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
        bias_init: initializer for the bias parameters (default: zeros_init()).
        carry_init: initializer for the internal state vector (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).

        init_method: parameter initialisation method to choose from. No other methods are 
            currently supported for the scalable REN (TODO). Options are:
        
        - "random" (default): Random sampling with `recurrent_kernel_init`.
        - "random_explicit": Randomly sample explicit model, not direct params.
        - "long_memory_explicit": Long-term init of explicit model not direct params.
        
        init_output_zero: initialize the network so its output is zero (default: False).
        identity_output: enforce that output layer is ``y_t = x_t``. (default: False).
        
        init_as_linear: Tuple of (A, B, C, D) matrices to initialise the contracting REN 
            as a linear system. The linear system must be stable. Default is `()`, which 
            defaults back to `init_method`.
        explicit_init: initialise the REN from some ExplicitSRENParams (default: None).
            If this is not `None`, it supercedes all other initialisation options.
            
        do_polar_param: Use the polar parameterization for the H matrix (default: True).
        eps: regularising parameter for positive-definite matrices (default: machine 
            precision for `jnp.float32`).
        seed: random seed for randomly initialising explicit model (default: 0). 
        
    Example usage:

        >>> import jax, jax.numpy as jnp
        >>> from robustnn import scalable_ren as sren
        
        >>> rng = jax.random.key(0)
        >>> key1, key2 = jax.random.split(rng)

        >>> nu, nx, nv, ny = 1, 2, 4, 1
        >>> nh = (2, 4)
        >>> model = sren.ScalableREN(nu, nx, nv, ny, nh)
        
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
    bias_init: Initializer = init.zeros_init()
    carry_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    
    init_method: str = "random"
    init_output_zero: bool = False
    identity_output: bool = False
    
    init_as_linear: Tuple = ()
    explicit_init: ExplicitSRENParams = None
    _direct_explicit_init: DirectSRENParams = None
    
    do_polar_param: bool = True
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps # type: ignore
    seed: int = 0
    _gamma: jnp.float32 = 1.0 # type: ignore
    
    def setup(self):
        """Initialise the scalable REN direct params."""
        
        self._init_params()
        
    def __call__(self, state: Array, inputs: Array) -> Tuple[Array, Array]:
        """Call a scalable REN model

        Args:
            state (Array): internal model state.
            inputs (Array): model inputs.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        
        explicit = self._direct_to_explicit()
        return self._explicit_call(state, inputs, explicit)
        
    def _explicit_call(
        self, x: Array, u: Array, e: ExplicitSRENParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for a scalable REN.

        Args:
            x (Array): internal model state.
            u (Array): model inputs.
            e (ExplicitSRENParams): explicit params.

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
        """Simulate a scalable REN over a sequence of inputs.

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
        """Initialise the scalable REN state (carry).

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
        
    def _direct_to_explicit(self) -> ExplicitSRENParams:
        """Convert from direct to explicit scalable REN params.

        Args:
            None

        Returns:
            ExplicitSRENParams: explicit params for scalable REN.
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
        
        return ExplicitSRENParams(
            A, B1, ps.B2, ps.C1, ps.C2, ps.D12, ps.D21, ps.D22, ps.bx, ps.bv, ps.by,
            network_params = self.network._direct_to_explicit()
        )
            
    def _x_to_h_contracting(self, X: Array, p: Array, B1: Array, C1: Array) -> Array:
        """Convert scalable REN X matrix to part of H matrix used in the contraction
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
    
    
    #################### Initialization Functions ####################
    
    def _init_params(self):
        """
        High-level init wrapper to initialise direct params either randomly
        or from an explicit model.
        """
        # Check if the user has followed instructions
        if self._check_do_explicit_init():
            if self._direct_explicit_init is None:
                raise ValueError(
                    "You have chosen to init a REN from an explicit model but have " +
                    "not called the `explicit_pre_init` method yet. Call this before " + 
                    "calling the typical `model.init()` and/or `model.apply()` " +
                    "methods in Flax."
                )
        
        if self.init_method not in get_valid_init():
            raise ValueError("Undefined init method '{}'".format(self.init_method))
        
        # Run the appropriate init
        if self._check_do_explicit_init():
            self._init_from_explicit()
        else:
            self._init_params_direct()
    
    def _network_init(self):
        """Initialise the LBDN for the equilibrium layer"""
        self.network = lbdn.LBDN(
            input_size=self.features,
            hidden_sizes=self.hidden,
            output_size=self.features,
            gamma=self._gamma,
            activation=self.activation,
            kernel_init=self.kernel_init,
        )
        
    def _init_params_direct(self):
        """Initialise all direct params for a scalable REN and store."""
        
        if self.init_method not in get_valid_init():
            raise ValueError("Undefined init method '{}'".format(self.init_method))
        
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        dtype = self.param_dtype
        
        # Initialise an LBDN for the equilibrium layer
        self._network_init()
        
        # Initialise free parameters        
        B2 = self.param("B2", self.kernel_init, (nx, nu), dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), dtype)
        bx = self.param("bx", self.bias_init, (nx,), dtype)
        bv = self.param("bv", self.bias_init, (nv,), dtype)
        
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
            out_bias_init = self.bias_init
        
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
            
        self.direct = DirectSRENParams(
            p, X, Y, B1, B2, C1, D12, C2, D21, D22, bx, bv, by, self.network.direct
        )
        
    def _x_long_memory_init(self):
        """Initialise the X matrix so A is close to the identity.
        
        Assumes B1, C1 = 0 and Y = E = I.
        """
        def init_func(key, shape, dtype) -> Array:
            nx = self.state_size
            dtype = self.param_dtype
            
            E = jnp.identity(nx, dtype)
            A = jnp.identity(nx, dtype)
            P = jnp.identity(nx, dtype)
            
            H = jnp.block([
                [(E + E.T - P), A.T],
                [A, P]
            ]) + self.eps * jnp.identity(shape[0])
            
            X = jnp.linalg.cholesky(H, upper=True)
            return X
        
        return init_func
    
    
    #################### Explicit Initialisation ####################
    
    def _check_do_explicit_init(self):
        return ((self.explicit_init is not None) or 
                ("explicit" in self.init_method))
    
    def explicit_pre_init(self):
        """A non-jittable method allowing initialisation from an explicit model.
        
        Call this before running `model.init()`, `model.apply()`, or anything else
        if you want to initialise the scalable REN from an explicit model. This is
        to avoid having non-jittable code in the `setup()` or `__call__()` methods.
        """
        # Can initialise from linear model if needed
        if self.init_as_linear:
            self._init_linear_sys()
        
        # Skip if not required
        if not self._check_do_explicit_init():
            return None
        
        # Get and check an explicit model
        explicit = self.explicit_init
        if explicit is None:
            explicit = self._generate_explicit_params()
            
        # Compute direct params reproducing this explicit model and store
        direct = self._explicit_to_direct(explicit)
        self._direct_explicit_init = direct
        
    def _init_linear_sys(self):
        """Initialise the scalable contracting REN as a (stable) linear system."""
        
        # Extract params and system sizes
        A, B, C, D = self.init_as_linear
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        dtype = self.param_dtype
        
        # Error checking
        nx_a = A.shape[0]
        assert (A.shape[0] <= nx and A.shape[1] == A.shape[0])
        assert B.shape == (nx_a, nu)
        assert C.shape == (ny, nx_a)
        assert D.shape == (ny, nu)
        
        # Fill out A matrix to match the required number of states
        dnx = nx - nx_a
        A = jnp.block([
            [A, jnp.zeros((nx_a, dnx), dtype)],
            [jnp.zeros((dnx, nx_a), dtype), jnp.zeros((dnx, dnx), dtype)],
        ])
        B = jnp.vstack([B, jnp.zeros((dnx, nu), dtype)])
        C = jnp.hstack([C, jnp.zeros((ny, dnx), dtype)])
        
        # Set up an explicit model to initialise from in the pre-init
        explicit = ExplicitSRENParams(
            A = A,
            B1 = jnp.zeros((nx_a, nv), dtype),
            B2 = B,
            C1 = jnp.zeros((nv, nx_a), dtype),
            C2 = C,
            D12 = jnp.zeros((nv, nu), dtype),
            D21 = jnp.zeros((ny, nv), dtype),
            D22 = D,
            bx = jnp.zeros((nx,), dtype),
            bv = jnp.zeros((nv,), dtype),
            by = jnp.zeros((ny,), dtype),
            network_params=None
        )
        self.explicit_init = explicit
    
    def _generate_explicit_params(self):
        """Randomly generate explicit parameterisation for a scalable REN."""
        # Sizes and dtype
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        dtype = self.param_dtype
        
        # Random seed
        rng = jax.random.key(self.seed)
        keys = jax.random.split(rng, 11)
        
        # Get orthogonal/diagonal matrices for a stable A-matrix
        if self.init_method == "random_explicit":
            D = jax.random.uniform(keys[0], nx)
        elif self.init_method == "long_memory_explicit":
            D = 1 - 0.01*jax.random.uniform(keys[0], nx)
        U = init.orthogonal()(keys[1], (nx,nx), dtype)
        V = init.orthogonal()(keys[2], (nx,nx), dtype)
        
        # State and equilibrium layers
        # At the moment, B1, C1 will be chosen so that 
        # the contraction LMI is feasible, so they are actually ignored. 
        A = V @ jnp.diag(D) @ U.T
        B1 = self.kernel_init(keys[3], (nx, nv), dtype)
        B2 = self.kernel_init(keys[4], (nx, nu), dtype)
        bx = self.bias_init(keys[5], (nx,), dtype)
        
        C1 = self.kernel_init(keys[0], (nv, nx), dtype)
        D12 = self.kernel_init(keys[6], (nv, nu), dtype)
        bv = self.bias_init(keys[7], (nv,), dtype)
        
        # Choose output layer specially
        if self.init_output_zero:
            out_kernel_init = init.zeros_init()
            out_bias_init = init.zeros_init()
        else:
            out_kernel_init = self.kernel_init
            out_bias_init = self.bias_init
            
        if self.identity_output:
            C2 = jnp.identity(nx)
            D21 = jnp.zeros((ny, nv), dtype)
            by = jnp.zeros((ny,), dtype)
        else:
            by = out_bias_init(keys[8], (ny,), dtype)
            C2 = out_kernel_init(keys[9], (ny, nx), dtype)
            D21 = out_kernel_init(keys[10], (ny, nv), dtype)    
        D22 = jnp.zeros((ny, nu), dtype)
        
        # Randomly generated explicit params
        return ExplicitSRENParams(
            A, B1, B2, C1, C2, D12, D21, D22, bx, bv, by, network_params=None
        )
    
    def _explicit_to_direct(self, e: ExplicitSRENParams) -> DirectSRENParams:
        """Find direct scalable REN parameterisation that admits the given explicit params.

        Args:
            e (ExplicitSRENParams): Explicit scalable REN params (e.g. from init).

        Returns:
            DirectSRENParams: Direct scalable REN params (these are learnable).
        """
        nx = self.state_size
        nv = self.features
        dtype = self.param_dtype
        
        # # Create variables
        # P = cp.Variable((nx, nx), symmetric=True)
        # C1 = cp.Variable((nv, nx))
        # B1 = cp.Variable((nx, nv))
        # A = e.A
        
        # # Generate constraints for contraction and small-gain
        # # TODO: At the moment this results in B1, C1 = 0. Fix later.
        # lhs = cp.bmat([
        #     [jnp.identity(nv), C1, jnp.zeros((nv, nv + nx))],
        #     [C1.T, P, jnp.zeros((nx, nv)), A.T],
        #     [jnp.zeros((nv, nv + nx)), jnp.identity(nv), B1.T],
        #     [jnp.zeros((nx, nv)), A, B1, 2*jnp.identity(nx) - P]
        # ])
        # constraints = [
        #     lhs >> 0,
        #     P >> np.identity(nx)
        # ]
        
        # # Solve SDP
        # prob = cp.Problem(cp.Minimize(cp.norm(P)), constraints)
        # prob.solve(solver=cp.MOSEK)
        # if prob.status != cp.OPTIMAL:
        #     print("Problem status: ", prob.status)
        #     raise ValueError("Could not find valid P, Lambda for explicit params.")
        
        # P = jnp.array(P.value)
        # C1 = jnp.array(C1.value)
        # B1 = jnp.array(B1.value)
        
        # TODO: SDP approach currently not working. Just assume B1, C1 = 0
        # for now and solve P - AT @ P @ A >= 0.
        B1 = jnp.zeros((nx, nv), dtype)
        C1 = jnp.zeros((nv, nx), dtype)
        P = solve_discrete_lyapunov_direct(e.A.T, jnp.identity(nx))
        
        # Compute the implicit model params
        E = P
        A_imp = E @ e.A
        B1_imp = E @ B1
        
        Hdiff = jnp.block([
            [E.T + E - P, A_imp.T],
            [A_imp, P]
        ]) - jnp.block([
            [C1.T @ C1, jnp.zeros((nx, nx))],
            [jnp.zeros((nx, nx)), B1_imp @ B1_imp.T]
        ]) + self.eps * jnp.identity(2*nx)
        X = jnp.linalg.cholesky(Hdiff, upper=True)
        
        return DirectSRENParams(
            p = l2_norm(X, eps=self.eps),
            X = X,
            Y = E,
            B1 = B1_imp, 
            B2 = e.B2,
            C1 = C1,
            D12 = e.D12,
            C2 = e.C2,
            D21 = e.D21,
            D22 = e.D22,
            bx = e.bx,
            bv = e.bv,
            by = e.by,
            network_params=None
        )
        
    def _init_from_explicit(self):
        """Initialise direct params from an existing explicit scalable REN model.
        
        This method requires the `explicit_pre_init` method to have been
        called first to correctly populate the `_direct_explicit_init` field.
        """      
        # Set up all the LTI parameters
        direct = self._direct_explicit_init
        dtype = self.param_dtype
        ps = {}
        for field in direct.__dataclass_fields__:
            if field == "network_params": continue
            val = getattr(direct, field)
            ps[field] = self.param(field, init.constant(val), val.shape, dtype)
            
        # Initialise network and store
        self._network_init()
        self.direct = DirectSRENParams(**ps, network_params=self.network.direct)

    def _check_valid_explicit(self, e: ExplicitSRENParams):
        """Error checking to help with explicit init."""
        
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        
        assert e.A.shape == (nx, nx)
        assert e.B1.shape == (nx, nv)
        assert e.B2.shape == (nx, nu)
        
        assert e.C1.shape == (nv, nx)
        assert e.D12.shape == (nv, nu)
        
        assert e.C2.shape == (ny, nx)
        assert e.D21.shape == (ny, nv)
        assert e.D22.shape == (ny, nu)
        
        assert e.bx.shape == (nx,)
        assert e.bv.shape == (nv,)
        assert e.by.shape == (ny,)
        
        # A matrix must be stable for contraction
        eig_norms = jnp.abs(jnp.linalg.eigvals(e.A))
        assert jnp.all(eig_norms < 1.0)
        
        
    #################### Convenient Wrappers ####################

    def explicit_call(
        self, params:dict, x: Array, u: Array, e: ExplicitSRENParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for a scalable REN.

        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): internal model state.
            u (Array): model inputs.
            e (ExplicitSRENParams): explicit params.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        return self.apply(params, x, u, e, method="_explicit_call")
    
    def simulate_sequence(self, params: dict, x0, u) -> Tuple[Array, Array]:
        """Simulate a scalable REN over a sequence of inputs.

        Args:
            params (dict): Flax model parameters dictionary.
            x0: array of initial states, shape is (batches, ...).
            u: array of inputs as a sequence, shape is (time, batches, ...).
            
        Returns:
            Tuple[Array, Array]: (final_state, outputs in (time, batches, ...)).
        """
        return self.apply(params, x0, u, method="_simulate_sequence")
    
    def direct_to_explicit(self, params: dict) -> ExplicitSRENParams:
        """Convert from direct to explicit scalable REN params.

        Args:
            params (dict): Flax model parameters dictionary.

        Returns:
            ExplicitSRENParams: explicit params for scalable REN.
        """
        return self.apply(params, method="_direct_to_explicit")
