import jax
import jax.numpy as jnp
import cvxpy as cp
import numpy as np

from functools import partial
from typing import Tuple

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array

from robustnn.utils import l2_norm, identity_init
from robustnn.utils import ActivationFn, Initializer


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
    
    
@dataclass
class ImplicitRENParams:
    """Data class to keep track of implicit params for a REN.
    
    These are the intermediate parameters used between the direct
    and explicit formulations.
    """
    E: Array
    F: Array
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


def get_valid_init():
    return ["random", "long_memory", "random_explicit", "long_memory_explicit"]


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
        
        kernel_init: initializer for weights (default: lecun_normal()).
        recurrent_kernel_init: initializer for the REN `X` matrix (default: lecun_normal()).
        bias_init: initializer for the bias parameters (default: zeros_init()).
        carry_init: initializer for the internal state vector (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        
        init_method: parameter initialisation method to choose from. Options are:
            - `"random"` (default): Random sampling with `recurrent_kernel_init`.
            - `"long_memory"`: Initialise such that `A = I` (approx.) in explicit model.
                Good for long-memory dynamics on initialisation.
            - `"random_explicit": Randomly sample explicit model, not direct params.
            - `"long_memory_explicit": Long-term init of explicit model not direct params.
        init_output_zero: initialize the network so its output is zero (default: False).
        identity_output: Exclude output layer ``y_t = C_2 x_t + D_{21} w_t + D_{22} u_t + 
            b_y``. Otherwise, output is just ``y_t = x_t``. (default: False).
        explicit_init: explicit REN parameters to use for initialization (default: None).
            If this is not `None`, it supercedes all other initialisation options.
            
        do_polar_param: Use the polar parameterization for the H matrix (default: True).
        d22_zero: Fix `D22 = 0` to remove any feedthrough in the REN (default: False).
        abar: upper bound on the contraction rate. Requires
            `0 <= abar <= 1` (default: 1).
        eps: Regularising parameter for positive-definite matrices (default: machine 
            precision for `jnp.float32`).
        seed: Random seed for initialising explicit model (default: 0). This is not a 
            nice way to handle explicit init. Make it use the random seed from model.init()
            instead in the future. (TODO: get rid of this?)
            
            https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/rng_guide.html#using-self-param-and-self-variable
    """
    input_size: int     # nu
    state_size: int     # nx
    features: int       # nv
    output_size: int    # ny
    activation: ActivationFn = nn.relu
    
    kernel_init: Initializer = init.lecun_normal()
    recurrent_kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()
    carry_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    
    init_method: str = "random"
    init_output_zero: bool = False
    identity_output: bool = False
    explicit_init: ExplicitRENParams = None
    
    do_polar_param: bool = True
    d22_zero: bool = False
    abar: jnp.float32 = 1 # type: ignore
    eps: jnp.float32 = jnp.finfo(jnp.float32).eps # type: ignore
    seed: int = 0
    
    def setup(self):

        # Error checking
        self._error_check_output_layer()
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
    
    def simulate_sequence(self, params, x0, u):
        """Simulate a REN over a sequence of inputs.

        Args:
            params: the usual model parameters dict.
            x0: array of initial states, shape is (batches, ...).
            u: array of inputs as a sequence, shape is (time, batches, ...).
            
        Returns:
            x1: internal state at the end of the sequence.
            y: array of outputs as a sequence, shape is (time, batches, ...).
            
        Note:
            - Use this if you would otherwise do `model.apply()` in a loop.
            - The direct -> explicit map is only called once, at the start
            of the sequence. This avoids unnecessary calls to the parameter
            mapping and should speed up your code :)
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
    
    def params_to_explicit(self, ps: dict):
        """
        Convert a parameter dictionary returned by the `.init()` method
        in Flax into an `ExplicitRENParams` instance.
        
        The `ps` dictionary must be of the form (with possibly different sizes):
        {'params': {'B2': (2, 1), 'C2': (1, 2), 'D12': (4, 1), 'D21': (1, 4), 'D22': (1, 1), 'X': (8, 8), 'X3': (1, 1), 'Y1': (2, 2), 'Y3': (1, 1), 'Z3': (0, 1), 'bv': (1, 4), 'bx': (1, 2), 'by': (1, 1), 'polar': (1,)}}
        """
        
        # Special handling for identity output layer for now.
        # Should make this more streamlined (TODO)
        if self.identity_output:
            nu = self.input_size
            nx = self.state_size
            nv = self.features
            ny = self.output_size
            C2 = jnp.identity(nx)
            D21 = jnp.zeros((ny, nv), self.param_dtype)
            D22 = jnp.zeros((ny, nu), self.param_dtype)
            by = jnp.zeros((ny,), self.param_dtype)
        else:
            C2 = ps["params"]["C2"]
            D21 = ps["params"]["D21"]
            D22 = ps["params"]["D22"]
            by = ps["params"]["by"]
        
        direct = DirectRENParams(
            p = ps["params"]["p"],
            X = ps["params"]["X"],
            B2 = ps["params"]["B2"],
            D12 = ps["params"]["D12"],
            Y1 = ps["params"]["Y1"],
            X3 = ps["params"]["X3"],
            Y3 = ps["params"]["Y3"],
            Z3 = ps["params"]["Z3"],
            bx = ps["params"]["bx"],
            bv = ps["params"]["bv"],
            C2 = C2,
            D21 = D21,
            D22 = D22,
            by = by,
        )
        return self._direct_to_explicit(direct)
    
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
        
        # Biases can go unchanged
        bx_e = ps.bx
        bv_e = ps.bv
        by_e = ps.by
        
        # Remaining explicit params are biases/in the output layer (unchanged)
        explicit = ExplicitRENParams(A_e, B1_e, B2_e, C1_e, ps.C2, D11_e, 
                                     D12_e, ps.D21, D22, bx_e, bv_e, by_e)
        return explicit
    
    def _x_to_h_contracting(self, X: Array, p: Array) -> Array:
        """
        Convert REN X matrix to part of H matrix used in the contraction
        setup (using polar parameterization if required).
        """
        H = X.T @ X
        if self.do_polar_param:
            H = p**2 * H / (l2_norm(X)**2)
        return H + self.eps * jnp.identity(jnp.shape(X)[0])
    
    
    #################### Initialization Functions ####################
    
    def _init_params(self):
        
        # Initialise from explicit if provided
        if self.explicit_init is not None:
            self._init_from_explicit(self.explicit_init)
            return None
        
        # Error checking
        if self.init_method not in get_valid_init():
            raise ValueError("Undefined init method '{}'".format(self.init_method))
        
        # Either initialise direct params straight away, or initialise an
        # explicit model and compute the corresponding direct params
        if "explicit" in self.init_method:
            explicit = self._generate_explicit_params()
            self._init_from_explicit(explicit)
        else:
            self._init_params_direct()
        
    def _init_params_direct(self):
        """Initialise all direct params for a REN and store."""
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        dtype = self.param_dtype
        
        # Define direct params for REN
        B2 = self.param("B2", self.kernel_init, (nx, nu), dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), dtype)
        
        if self.init_method == "random":
            x_init = self.recurrent_kernel_init
        elif self.init_method == "long_memory":
            x_init = self._x_long_memory_init(B2, D12)
        X = self.param("X", x_init, (2*nx + nv, 2*nx + nv), dtype)
        
        p = self.param("p", init.constant(l2_norm(X, eps=self.eps)), (1,), dtype)
        Y1 = self.param("Y1", self.kernel_init, (nx, nx), dtype)
        
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
        
    def _x_long_memory_init(self, B2, D12):
        """Initialise the X matrix so E, F, P (and therefore A) are I."""
        
        def init_func(key, shape, dtype) -> Array:
            dtype = self.param_dtype
            key, rng = jax.random.split(key, 2)
            
            nx = B2.shape[0]
            nv = D12.shape[0]
            
            E = jnp.identity(nx, dtype)
            F = jnp.identity(nx, dtype)
            P = jnp.identity(nx, dtype)
            
            B1 = jnp.zeros((nx, nv), dtype)
            C1 = jnp.zeros((nv, nx), dtype)
            D11 = self.kernel_init(rng, (nv, nv), dtype)
            
            # Need eigvals of Lambda large enough so that H22 is pos def
            eigs, _ = jnp.linalg.eigh(D11 + D11.T)
            Lambda = (jnp.max(eigs) / 2 + 1e-4) * jnp.identity(nv, dtype)
            H22 = 2*Lambda - D11 - D11.T
            
            H = jnp.block([
                [(E + E.T - P), -C1.T, F.T],
                [-C1, H22, B1.T],
                [F, B1, P]
            ]) + self.eps * jnp.identity(shape[0])
            
            X = self._h_contracting_to_x(H)
            return X
        
        return init_func
    
    def _init_from_explicit(self, explicit: ExplicitRENParams):
        """Initialise direct params from an existing explicit REN model."""
        
        # Double-check valid sizes
        self._check_explicit(explicit)
        
        # Compute direct params matching this explicit model
        direct = self._explicit_to_direct(explicit)
        dtype = self.param_dtype
        
        # Initialise learnable params as these values
        ps = {}
        for field in direct.__dataclass_fields__:           
            val = getattr(direct, field)
            if val is None:
                ps[field] = self.param(field, init.zeros_init(), (0,), dtype)
            else:
                ps[field] = self.param(field, init.constant(val), val.shape, dtype)
            
        # Store learnable params
        self.direct = DirectRENParams(**ps)
    
    def _explicit_to_direct(self, e: ExplicitRENParams) -> DirectRENParams:
        """Find direct REN parameterisation that admits the given explicit params.

        Args:
            e (ExplicitRENParams): Explicit REN params (e.g., from initialisation).

        Returns:
            DirectRENParams: Direct REN params (these are learnable).
        """
        
        nx = self.state_size
        nv = self.features
        
        # Create the variables
        P = cp.Variable((nx, nx), symmetric=True)
        Lambda = cp.diag(cp.Variable((nv,), nonneg=True))
        
        # Generate constraints
        lhs = self._explicit_to_sdp(e, P, Lambda)
        constraints = [
            lhs >> 0,
            P >> np.identity(nx)
        ]
        
        # Solve the SDP
        prob = cp.Problem(cp.Minimize(cp.norm(P)), constraints)
        prob.solve(solver=cp.MOSEK)
        if prob.status != cp.OPTIMAL:
            raise ValueError("Could not find valid P, Lambda for explicit params.")
        
        P = jnp.array(P.value)
        Lambda = jnp.array(Lambda.value)
        
        # Compute the implicit model params
        E = P
        F = E @ e.A
        B1_imp = E @ e.B1
        B2_imp = E @ e.B2
        
        C1_imp = Lambda @ e.C1
        D11_imp = Lambda @ e.D11
        D12_imp = Lambda @ e.D12
        
        C2_imp = e.C2
        D21_imp = e.D21
        D22_imp = e.D22
        
        bx_imp = e.bx
        bv_imp = e.bv
        by_imp = e.by
        
        implicit = ImplicitRENParams(
            E, F, B1_imp, B2_imp, C1_imp, C2_imp,
            D11_imp, D12_imp, D21_imp, D22_imp,
            bx_imp, bv_imp, by_imp
        )
        
        # Build the H matrix
        H11 = E + E.T - (P / self.abar**2)
        H22 = 2 * Lambda - D11_imp - D11_imp.T
        H33 = P
        H21 = -C1_imp
        H31 = F
        H32 = B1_imp
        H = jnp.block([
            [H11, H21.T, H31.T],
            [H21, H22, H32.T],
            [H31, H32, H33]
        ]) # TODO: Add eps * identity for numerical conditioning?
        
        # Convert to final direct parameters. This depends on the 
        # specific REN parameterisation.
        direct = self._hmatrix_to_direct(H, implicit)
        return direct
    
    def _h_contracting_to_x(self, H: Array) -> Array:
        """
        Convert part of H matrix used in the contraction setup
        to REN X matrix (if using polar parameterization, set p = norm(X)).
        """
        # TODO: Can we do something better than cholesky?
        return jnp.linalg.cholesky(H, upper=True)
    
    
    ############### Specify these for each REN parameterisation ###############
    
    def _error_checking(self):
        """Check conditions for REN."""
        raise NotImplementedError(
            "Each REN parameterisation should have its own version of this function."
        )
        
    def _direct_to_explicit(self, direct: DirectRENParams) -> ExplicitRENParams:
        """
        Convert direct paremeterization of a REN to explicit form
        for evaluation. This depends on the specific REN parameterization.
        """
        raise NotImplementedError(
            "RENBase models should not be called. " +
            "Choose a REN parameterization instead (eg: `ContractingREN`)."
        )
        
    def _generate_explicit_params(self):
        """Randomly generate explicit parameterisation for a REN.
        
        This method depends on the specific REN parameterisation.
        """
        raise NotImplementedError(
            "This REN parameterisation currently does not support initialisation " +
            "from a random explicit model."
        )
    
    def _explicit_to_sdp(self, e: ExplicitRENParams, P: cp.Variable, Lambda: cp.Variable):
        """Set up the LHS of the SDP used to solve for P, Lambda using CVXPY.

        Args:
            e (ExplicitRENParams): Explicit REN params.
            P (cp.Variable): P matrix to solve for (pos def).
            Lambda (cp.Variable): Lambda matrix to solve for (nonneg, diag).

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(
            "This function is called within the `_explicit_to_direct` method. " +
            "Each REN parameterization will have a different implementation. See " +
            "Eqns. (15) - (16) of the REN paper."
        )
        
    def _hmatrix_to_direct(self, H: Array, implicit: ImplicitRENParams) -> DirectRENParams:
        """Compute direct params from REN H matrix and implicit params.

        Args:
            H (Array): The REN H matrix
            implicit (ImplicitRENParams): Implicit REN params.

        Returns:
            DirectRENParams: Direct REN params.
            
        Note:
            If any of the fields in `DirectRENParams` are not used by a specific
            parameterisation, just leave them as `None`. Y1 handled separately.
        """
        raise NotImplementedError(
            "This function is called within the `_explicit_to_direct` method. " +
            "Each REN parameterization will have a different implementation. See " +
            "Eqns. (21) & (29) of the REN paper."
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

    def _check_explicit(self, e: ExplicitRENParams):
        """Error checking to help with explicit init."""
        
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = self.output_size
        
        assert e.A.shape == (nx, nx)
        assert e.B1.shape == (nx, nv)
        assert e.B2.shape == (nx, nu)
        
        assert e.C1.shape == (nv, nx)
        assert e.D11.shape == (nv, nv)
        assert e.D12.shape == (nv, nu)
        
        assert e.C2.shape == (ny, nx)
        assert e.D21.shape == (ny, nv)
        assert e.D22.shape == (ny, nu)
        
        assert e.bx.shape == (nx,)
        assert e.bv.shape == (nv,)
        assert e.by.shape == (ny,)
        
        # D11 must be lower triangular
        assert jnp.all(e.D11 == jnp.tril(e.D11, k=-1))
        
        # A matrix must be stable for contraction
        eig_norms = jnp.abs(jnp.linalg.eigvals(e.A))
        assert jnp.all(eig_norms < 1.0)
