# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Base class for linear Recurrent Equilibrium Networks (RENs).

See ./robustnn/ren_base.py for a description. These linear RENs have nv = 0.

Author: Nic Barbara.
'''

import jax
import jax.numpy as jnp

from typing import Tuple

from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Array

from robustnn.ren_base import RENBase
from robustnn.utils import l2_norm, identity_init


def get_valid_init():
    return ["random", "long_memory"]


@dataclass
class DirectLinRENParams:
    """Data class to keep track of direct params for a linear REN.
    
    These are the free, trainable params of the model. They are a subset
    of the usual REN parameters, where B = B2, C = C2, D = D22.
    """
    p: Array
    X: Array
    Y1: Array
    B: Array
    C: Array
    D: Array
    X3: Array
    Y3: Array
    Z3: Array
    bx: Array
    by: Array
    
@dataclass
class ExplicitLinRENParams:
    """Data class to keep track of explicit params for a Linear REN.
    
    These are the parameters used for evaluating the model. They are a subset
    of the usual REN parameters, where B = B2, C = C2, D = D22.
    """
    A: Array
    B: Array
    C: Array
    D: Array
    bx: Array
    by: Array
    

class LinRENBase(RENBase):
    """
    Base class for linear Recurrent Equilibrium Networks (RENs).
    
    `See RENBase` for a description of all relevant parameters. `LinRENBase` is to be
    used only for linear systems (when the number of neurons is zero in the REN) and
    differes from the `RENBase` construction as follows:
    
    - The number of `features` must always be zero. Initialise linear RENs with, for
      example, linren = ContractingLinREN(nu, nx, 0, ny).
      
    - There is no activation function. Setting it will do nothing.
    
    - `identity_output=true` is currently not supported (it's only supported for
      contracting nonlinear RENs at the moment anyway).
      
    Linear RENs will be useful for users who want to construct stable linear systems
    satisfying (Q, S, R)-dissipativity properties with a direct parametrisation.
    """
    
    def setup(self):
        """Initialise direct (implicit) params for a Linear REN."""
        
        # Error checking
        self._error_check_linear()
        self._error_check_output_layer()
        self._error_checking()
        if self.init_method not in get_valid_init():
            raise ValueError("Undefined init method '{}'".format(self.init_method))
        
        # Sizes
        nu = self.input_size
        nx = self.state_size
        ny = self.output_size
        dtype = self.param_dtype
        
        # Define direct params
        B = self.param("B", self.kernel_init, (nx, nu), dtype)
        bx = self.param("bx", self.x_bias_init, (nx,), dtype)
        
        if self.init_method == "random":
            x_init = self.recurrent_kernel_init
            Y1 = self.param("Y1", self.kernel_init, (nx, nx), dtype)
            
        elif self.init_method == "long_memory":
            x_init = self._x_long_memory_init()
            Y1 = self.param("Y1", init.constant(jnp.identity(nx)), (nx, nx), dtype)
            
        X = self.param("X", x_init, (2*nx, 2*nx), dtype)
        p = self.param("p", init.constant(l2_norm(X, eps=self.eps)), (1,), dtype)
        
        # Output layer params
        if self.init_output_zero:
            out_kernel_init = init.zeros_init()
            out_bias_init = init.zeros_init()
        else:
            out_kernel_init = self.kernel_init
            out_bias_init = self.y_bias_init
        
        C = self.param("C", out_kernel_init, (ny, nx), dtype)
        D = self.param("D", init.zeros_init(), (ny, nu), dtype)
        by = self.param("by", out_bias_init, (ny,), dtype)
        
        if self.d22_zero:
            D = jnp.zeros((ny, nu), dtype)
            
        # These parameters are used to construct D instead of the above for most RENs.
        # Could tidy up the code a little here by not initialising D at all.
        # By default they initialise D = 0
        d = min(nu, ny)
        X3 = self.param("X3", identity_init(), (d, d), dtype)
        Y3 = self.param("Y3", init.zeros_init(), (d, d), dtype)
        Z3 = self.param("Z3", init.zeros_init(), (abs(ny - nu), d), dtype)
        
        # Set up the direct parameter struct
        self.direct = DirectLinRENParams(p, X, Y1, B, C, D, X3, Y3, Z3, bx, by)
            
        
    def _x_long_memory_init(self):
        """Initialise the X matrix so A is close to the identity.
        
        Assumes Y = E = I.
        # TODO: Don't think it works when B, C != 0 too unless it's just
        # TODO: for a contracting linear REN. Same goes for regular REN.
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
    
    def __call__(self, state: Array, inputs: Array) -> Tuple[Array, Array]:
        """Call a linear REN model

        Args:
            state (Array): internal model state.
            inputs (Array): model inputs.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(state, inputs, explicit)
    
    def _explicit_call(
        self, x: Array, u: Array, e: ExplicitLinRENParams
    ) -> Tuple[Array, Array]:
        """Evaluate explicit model for a linear REN.

        Args:
            x (Array): internal model state.
            u (Array): model inputs.
            e (ExplicitLinRENParams): explicit params.

        Returns:
            Tuple[Array, Array]: (next_states, outputs).
        """
        x1 = x @ e.A.T + u @ e.B.T + e.bx
        y  = x @ e.C.T + u @ e.D.T + e.by
        return x1, y
    
    def _hmatrix_to_explicit(
        self, ps: DirectLinRENParams, H: Array, D: Array
    ) -> ExplicitLinRENParams:
        """Convert linear REN H matrix to explict model given direct params.

        Args:
            ps (DirectLinRENParams): direct linear REN params.
            H (Array): Linear REN H matrix used in parameterisation (see Eqns. 19, 29).
            D (Array): The D matrix to be used. Allows for special construction.

        Returns:
            ExplicitLinRENParams: explicit linear REN model.
        """
        nx = self.state_size
        
        # Extract sections of the H matrix
        H11 = H[:nx, :nx]
        H21 = H[nx:(2*nx), :nx]
        H22 = H[nx:(2*nx), nx:(2*nx)]
        
        # Construct implicit model parameters
        P_imp = H22
        F = H21
        E = (H11 + P_imp / (self.abar**2) + ps.Y1 - ps.Y1.T) / 2
        
        # Construct the explicit model
        A = jnp.linalg.solve(E, F)
        B = jnp.linalg.solve(E, ps.B)
        
        # Remaining explicit params are biases/in the output layer (unchanged)
        explicit = ExplicitLinRENParams(A, B, ps.C, D, ps.bx, ps.by)
        return explicit
    
    
    ############### Specify for each linear REN parameterisation ###############
    
    def _direct_to_explicit(self) -> ExplicitLinRENParams:
        """
        Convert direct paremeterization of a linear REN to explicit form
        for evaluation. This depends on the specific linear REN parameterization.
        """
        raise NotImplementedError(
            "LinRENBase models should not be called. " +
            "Choose a linear REN parameterization instead (eg: `ContractingLinREN`)."
        )
    
    
    #################### Error checking ####################
    
    def _error_check_linear(self):
        if self.features != 0:
            raise ValueError(
                "Linear REN must have number of features = 0. Use LinREN(nu, nx, 0, ny)"
            )
        if self.identity_output:
            raise NotImplementedError(
                "Identity output not currently supported for linear REN."
            )
    