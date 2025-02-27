import jax
import jax.numpy as jnp

from typing import Tuple, Sequence

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array

from robustnn import lbdn
from robustnn.utils import l2_norm, cayley
from robustnn.utils import ActivationFn, Initializer


def get_valid_init():
    return ["random"]


@dataclass
class DirectSRENParams:
    """Data class to keep track of direct params for Scalable REN.
    
    These are the free, trainable parameters for a Scalable REN,
    excluding those in the LBDN layer.
    """
    p1: Array
    p2: Array
    p3: Array
    Xbar: Array
    Y1: Array
    Y2: Array
    Y3: Array
    B2: Array
    D12: Array
    C2: Array
    D21: Array
    D22: Array
    bx: Array
    bv: Array
    by: Array


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
        recurrent_kernel_init: currently unused (default: lecun_normal()).
        bias_init: initializer for the bias parameters (default: zeros_init()).
        carry_init: initializer for the internal state vector (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).

        init_method: parameter initialisation method to choose from. No other methods are 
            currently supported for the scalable REN (TODO). Options are:
        
        - "random" (default): Random sampling with `recurrent_kernel_init`.
        
        init_output_zero: initialize the network so its output is zero (default: False).
        identity_output: enforce that output layer is ``y_t = x_t``. (default: False).
        eps: regularising parameter for positive-definite matrices (default: machine 
            precision for `jnp.float32`).
            
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
    
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps # type: ignore
    _gamma: jnp.float32 = 1.0 # type: ignore
    
    def setup(self):
        """Initialise the scalable REN direct params."""
        
        if self.init_method not in get_valid_init():
            raise ValueError("Undefined init method '{}'".format(self.init_method))
        
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        nh = self.hidden
        dtype = self.param_dtype
        
        # Initialise an LBDN for the equilibrium layer
        self.network = lbdn.LBDN(
            input_size=nv,
            hidden_sizes=nh,
            output_size=nv,
            gamma=self._gamma,
            activation=self.activation,
            kernel_init=self.kernel_init,
        )
        
        # Most of these matrices have underspecified number of cols. Fix
        # it as smallest of (nx, nv) for now.
        d = min(nx, nv)
        n1 = nv         # To make X11 square (Cayley)
        n2 = d
        n3 = nv         # To make X43 square (Cayley)
        n4 = d
        
        # Initialise remaining free parameters
        Xbar = self.param("Xbar", self.kernel_init, (nx, n1 + 2*n2 + n3), dtype)
        p1 = self.param("p1", init.constant(l2_norm(Xbar, eps=self.eps)), (1,), dtype)
        Y1 = self.param("Y1", self.kernel_init, (nx, nx), dtype)
        
        Y2 = self.param("Y2", self.kernel_init, (n1, nv), dtype)
        p2 = self.param("p2", init.constant(l2_norm(Y2, eps=self.eps)), (1,), dtype)
        
        Y3 = self.param("Y3", self.kernel_init, (n3 + n4, nv), dtype)
        p3 = self.param("p3", init.constant(l2_norm(Y3, eps=self.eps)), (1,), dtype)
        
        B2 = self.param("B2", self.kernel_init, (nx, nu), dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), dtype)
        bx = self.param("bx", self.bias_init, (nx,), dtype)
        bv = self.param("bv", self.bias_init, (nv,), dtype)
        
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
            
        self.direct = DirectSRENParams(p1, p2, p3, Xbar, Y1, Y2, Y3, B2, 
                                       D12, C2, D21, D22, bx, bv, by)
        
    def __call__(self, state: Array, inputs: Array) -> Tuple[Array, Array]:
        """Call a scalable REN model

        Args:
            state (Array): internal model state.
            inputs (Array): model inputs.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        
        direct = (self.direct, self.network._get_direct_params())
        explicit = self._direct_to_explicit(direct)
        return self.explicit_call(state, inputs, explicit)
    
    def explicit_call(
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
        w = lbdn.LBDN._explicit_call(
            v, e.network_params, self._gamma, self.activation
        )
        
        # State-space model
        x1 = x @ e.A.T + w @ e.B1.T + u @ e.B2.T + e.bx
        y = x @ e.C2.T + w @ e.D21.T + u @ e.D22.T + e.by
        return x1, y
    
    def simulate_sequence(self, params, x0, u) -> Tuple[Array, Array]:
        """Simulate a scalable REN over a sequence of inputs.

        Args:
            params: the usual model parameters dict.
            x0: array of initial states, shape is (batches, ...).
            u: array of inputs as a sequence, shape is (time, batches, ...).
            
        Returns:
            Tuple[Array, Array]: (final_state, outputs in (time, batches, ...)).
        """
        explicit = self.params_to_explicit(params)
        def rollout(carry, ut):
            xt, = carry
            xt1, yt = self.explicit_call(xt, ut, explicit)
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

    def params_to_explicit(self, ps: dict) -> ExplicitSRENParams:
        """Convert from Flax params dict to explicit scalable REN params.

        Args:
            ps (dict): Flax params dict `{"params": {<model_params>}}`.

        Returns:
            ExplicitSRENParams: explicit params for scalable REN.
        """
        
        # Special handling for the output layer
        if self.identity_output:
            dtype = self.param_dtype
            C2 = jnp.identity(self.state_size, dtype)
            D21 = jnp.zeros((self.output_size, self.features), dtype)
            D22 = jnp.zeros((self.output_size, self.input_size), dtype)
            by = jnp.zeros((self.output_size,), dtype)
        else:
            C2 = ps["params"]["C2"]
            D21 = ps["params"]["D21"]
            D22 = ps["params"]["D22"]
            by = ps["params"]["by"]
        
        # Direct params for linear part of SREN model
        direct = DirectSRENParams(
            p1 = ps["params"]["p1"],
            p2 = ps["params"]["p2"],
            p3 = ps["params"]["p3"],
            Xbar = ps["params"]["Xbar"],
            Y1 = ps["params"]["Y1"],
            Y2 = ps["params"]["Y2"],
            Y3 = ps["params"]["Y3"],
            B2 = ps["params"]["B2"],
            D12 = ps["params"]["D12"],
            bx = ps["params"]["bx"],
            bv = ps["params"]["bv"],
            C2 = C2,
            D21 = D21,
            D22 = D22,
            by = by,
        )
        explicit = self._direct_to_explicit((direct, None))
        
        # Handle direct-to-explicit conversion for LBDN layer separately
        # Note that we cannot access self.network (or self.direct) outside
        # of the `setup()` and `__call__()` functions because Flax is a pain :(
        return explicit.replace(network_params = lbdn.LBDN.params_to_explicit(
            {"params": ps["params"]["network"]}
        ))
    
    def _direct_to_explicit(
        self, params: Tuple[DirectSRENParams, lbdn.DirectLBDNParams]
    ) -> ExplicitSRENParams:
        """Convert from direct scalable REN params to explicit params.

        Args:
            params (Tuple[DirectSRENParams, lbdn.DirectLBDNParams]): tuple of 
                direct params for the linear part of the scalable REN and
                the 1-Lipschitz network, respectively.

        Returns:
            ExplicitSRENParams: explicit scalable REN model.
            
        Leave the DirectLBDNParams as `None` if they are to be computed
        externally. This can be useful if converting straight from params dict.
        """
        
        ps, ps_network = params
        
        # Get all elements of the banded X-matrix first
        X_e = ps.p1 * ps.Xbar / l2_norm(ps.Xbar)
        X11_T = cayley(ps.p2 * ps.Y2 / l2_norm(ps.Y2))
        X43_T, X44_T = cayley(ps.p3 * ps.Y3 / l2_norm(ps.Y3), return_split=True)
        X11, X43, X44 = X11_T.T, X43_T.T, X44_T.T
        
        # Some shapes
        n1 = X11.shape[1]
        n3 = X43.shape[1]
        n4 = X44.shape[1]
        n2 = X_e.shape[1] - (n1 + n4 + n3)
        
        # Split them up into the components we need
        X21 = X_e[:, :n1]
        X22 = X_e[:, n1:(n1+n2)]
        X32 = X_e[:, (n1+n2):(n1+2*n2)]
        X33 = X_e[:, (n1+2*n2):]
        
        # Compute the implicit params
        E = (X_e @ X_e.T + ps.Y1 - ps.Y1.T) / 2
        A_imp = X32 @ X22.T
        B1_imp = X33 @ X43.T
        C1_imp = X11 @ X21.T
        
        # Explicit params for the network in feedback
        if ps_network is not None:
            network_explicit = lbdn.LBDN._direct_to_explicit(ps_network)
        else:
            network_explicit = None
        
        return ExplicitSRENParams(
            A = jnp.linalg.solve(E, A_imp),
            B1 = jnp.linalg.solve(E, B1_imp),
            B2 = ps.B2,
            C1 = C1_imp,
            D12 = ps.D12,
            C2 = ps.C2,
            D21 = ps.D21,
            D22 = ps.D22,
            bx = ps.bx,
            bv = ps.bv,
            by = ps.by,
            network_params = network_explicit
        )
            
    
    #################### Compatibility with RENs ####################
    
    def explicit_pre_init(self):
        """The RENs have this method. This way we can use the same
        high-level code."""
        pass
