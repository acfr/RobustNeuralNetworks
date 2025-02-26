import jax
import jax.numpy as jnp

from typing import Tuple

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array

from robustnn import lbdn
from robustnn.utils import l2_norm
from robustnn.utils import ActivationFn, Initializer


@dataclass
class DirectSRENParams:
    """Data class to keep track of direct params for Scalable REN.
    
    These are the free, trainable parameters for a Scalable REN.
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
    """TODO: Documentation everywhere.
    
    TODO: Consider inheriting from REN base class?
    """
    
    input_size: int     # nu
    state_size: int     # nx
    features: int       # nv
    output_size: int    # ny
    hidden: Tuple       # Hidden layer sizes in the LBDN
    activation: ActivationFn = nn.relu
    
    kernel_init: Initializer = init.lecun_normal()
    recurrent_kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()
    carry_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    
    init_output_zero: bool = False
    identity_output: bool = False
    # do_polar_param: bool = True # TODO: add support?
    
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps # type: ignore
    _gamma: jnp.float32 = 1.0 # type: ignore
    
    def setup(self):
        
        self._init_params()
        
    def __call__(self, state: Array, inputs: Array) -> Tuple[Array, Array]:
        
        direct = (self.direct, self.network._get_direct_params())
        explicit = self._direct_to_explicit(direct)
        return self.explicit_call(state, inputs, explicit)
    
    def explicit_call(
        self, x: Array, u: Array, e: ExplicitSRENParams
    ) -> Tuple[Array, Array]:
        """TODO: Split LBDN up into direct/explicit call too!"""
        
        # Equilibirum layer
        v = x @ e.C1.T + u @ e.D12.T + e.bv
        w = lbdn.LBDN._explicit_call(
            v, e.network_params, self._gamma, self.activation
        )
        
        # State-space model
        x1 = x @ e.A.T + w @ e.B1.T + u @ e.B2.T + e.bx
        y = x @ e.C2.T + w @ e.D21.T + u @ e.D22.T + e.by
        return x1, y
    
    def simulate_sequence(self, params, x0, u):
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
        batch_dims = input_shape[:-1]
        rng, _ = jax.random.split(rng)
        mem_shape = batch_dims + (self.state_size,)
        return self.carry_init(rng, mem_shape, self.param_dtype)

    def params_to_explicit(self, ps: dict):
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
            C2 = ps["params"]["C2"],
            D21 = ps["params"]["D21"],
            D22 = ps["params"]["D22"],
            bx = ps["params"]["bx"],
            bv = ps["params"]["bv"],
            by = ps["params"]["by"],
        )
        explicit = self._direct_to_explicit((direct, None))
        explicit = explicit.replace(network_params = lbdn.LBDN.params_to_explicit(
            {"params": ps["params"]["network"]}
        ))
        return explicit
    
    def _direct_to_explicit(
        self, params: Tuple[DirectSRENParams, lbdn.DirectLBDNParams]
    ) -> ExplicitSRENParams:
        """TODO: Split this up for contracting/other Scalable RENs"""
        
        ps, ps_network = params
        
        # Get all elements of the banded X-matrix first
        X_e = ps.p1 * ps.Xbar / l2_norm(ps.Xbar)
        X11_T = lbdn.cayley(ps.p2 * ps.Y2 / l2_norm(ps.Y2))
        X43_T, X44_T = lbdn.cayley(ps.p3 * ps.Y3 / l2_norm(ps.Y3), return_split=True)
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
        # P_imp = X32 @ X32.T + X33 @ X33.T
        
        # Explicit params for the network in feedback
        # They'll be none if we're reading straight from the params dict
        # TODO: This is a bit gross, can we tidy it up?
        if ps_network is not None:
            network_explicit = lbdn.LBDN._direct_to_explicit(ps_network)
        else:
            network_explicit = None
        
        # Explicit model
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
        
    
    #################### Initialization Functions ####################
    
    def _init_params(self):
        self._init_params_direct()
        
    def _init_params_direct(self):
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
            
    
    #################### Compatibility with RENs ####################
    
    def explicit_pre_init(self):
        pass
