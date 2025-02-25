'''
Implementation of Lipschitz Bounded Deep Networks (Linear) in JAX/FLAX

Adapted from Julia implentation: https://github.com/acfr/RobustNeuralNetworks.jl

Author: Jack Naylor, ACFR, Sep '23
Edited: Nic Barbara, ACFR, Mar '24

These networks should be compatible with other FLAX modules.
'''

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen import initializers as init
from jax import lax
from typing import Sequence, Optional
from flax.typing import Dtype, PrecisionLike

from robustnn.utils import l2_norm, ActivationFn, Initializer


def cayley(W, return_split=False):
    """Perform Cayley transform on a stacked matrix [U; V]"""
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    U, V = W[:n, :], W[n:, :]
    Z = (U - U.T) + (V.T @ V)
    I = jnp.eye(n)
    ZI = Z + I
    
    # Note that B * A^-1 = solve(A.T, B.T).T
    A_T = jnp.linalg.solve(ZI, I-Z)
    B_T = -2 * jnp.linalg.solve(ZI.T, V.T).T
    
    if return_split:
        return A_T, B_T
    return jnp.concatenate([A_T, B_T])


def dot_lax(input1, input2, precision: PrecisionLike = None):
    """
    Wrapper around lax.dot_general(). Use this instead of `@` for
    more accurate array-matrix multiplication (higher default precision?)
    """
    return lax.dot_general(
        input1,
        input2,
        (((input1.ndim - 1,), (1,)), ((), ())),
        precision=precision,
    )

class SandwichLayer(nn.Module):
    """A version of linen.Dense with a Lipschitz bound of 1.0.

    Example usage::

        >>> from robustnn.networks.lbdn import SandwichLayer
        >>> import jax, jax.numpy as jnp

        >>> layer = SandwichLayer(features=4)
        >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
        >>> jax.tree_map(jnp.shape, params)
        {'params': {'XY': (7, 4), 'a': (1,), 'b': (4,), 'd': (4,)}}

    Attributes:
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).
        is_output: treat this as the output layer of an LBDN (default: False).
        activation: Activation function to use (default: relu).
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.
        kernel_init: initializer function for the weight matrix (default: glorot_normal()).
        bias_init: initializer function for the bias (default: zeros_init()).
        psi_init: initializer function for the activation scaling (default: zeros_init()).
    """
    features: int
    use_bias: bool = True
    is_output: bool = False
    
    activation: ActivationFn = nn.relu
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = init.glorot_normal()
    bias_init: Initializer = init.zeros_init()
    psi_init: Initializer = init.zeros_init()
    
    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        
        # Set up parameters
        xy = self.param("XY", self.kernel_init, 
                        (jnp.shape(inputs)[-1]+self.features, self.features), 
                        self.param_dtype)
        
        a = self.param('a', init.constant(l2_norm(xy)), (1,), self.param_dtype)
        
        if self.use_bias: 
            b = self.param('b', self.bias_init, (self.features,), self.param_dtype)
            
        if not self.is_output: 
            d = self.param('d', self.psi_init, (self.features,), self.param_dtype)
            
        # Computations
        A_T, B_T = cayley(a / l2_norm(xy) * xy, return_split=True)
        B = B_T.T
        
        # If just the output layer, return Bx + b (or just Bx if no bias)
        # Using lax.dot_general directly instead of `@` because `linen.Dense`
        # does it too - look into this later.
        if self.is_output:
            if self.use_bias:
                return dot_lax(inputs, B) + b
            else:
                return dot_lax(inputs, B)
                
        # Regular sandwich layer (clip d to avoid over/underflow)
        psi_d = jnp.exp(jnp.clip(d, a_min=-20.0, a_max=20.0))
        x = jnp.sqrt(2.0) * dot_lax(inputs, ((jnp.diag(1 / psi_d)) @ B))
        if self.use_bias: 
            x += b
        x = jnp.sqrt(2.0) * dot_lax(self.activation(x), (A_T * psi_d.T))
        
        return x


class LBDN(nn.Module):
    """Lipschitz-Bounded Deep Network.
    
    Example usage::
    
        >>> from robustnn.lbdn import LBDN
        >>> import jax, jax.numpy as jnp
        
        >>> nu, ny = 5, 2
        >>> layers = (8, 16, ny)
        >>> gamma = jnp.float32(10)
        
        >>> model = LBDN(layer_sizes=layers, gamma=gamma)
        >>> params = model.init(jax.random.key(0), jnp.ones((6,nu)))
        >>> jax.tree_map(jnp.shape, params)
        {'params': {'SandwichLayer_0': {'XY': (13, 8), 'a': (1,), 'b': (8,), 'd': (8,)}, 'SandwichLayer_1': {'XY': (24, 16), 'a': (1,), 'b': (16,), 'd': (16,)}, 'SandwichLayer_2': {'XY': (18, 2), 'a': (1,), 'b': (2,)}, 'ln_gamma': (1,)}}
    
    Attributes:
        layer_sizes: Tuple of hidden layer sizes and the output size.
        gamma: Upper bound on the Lipschitz constant (default: inf).
        activation: Activation function to use (default: relu).
        kernel_init: Initialisation function for matrics (default: glorot_normal).
        use_bias: Whether to use bias terms (default: True).
        trainable_lipschitz: Make the Lipschitz constant trainable (default: False).
        init_output_zero: initialize the network so its output is zero (default: False).
    
    Note: Only monotone activations will work. Currently only identity, relu, tanh
          are supported
    
    Note: Optional activation on final layer is not implemented yet.
    """
    
    layer_sizes: Sequence[int]
    gamma: jnp.float32 = 1.0 # type: ignore
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = init.glorot_normal()
    use_bias: bool = True
    trainable_lipschitz: bool = False
    init_output_zero: bool = False
    
    def setup(self):
        """Define some common sizes."""
        self.hidden_sizes = self.layer_sizes[:-1]
        self.output_size = self.layer_sizes[-1]
        
    @nn.compact
    def __call__(self, inputs : jnp.array) -> jnp.array:
        
        # Set up trainable/constant Lipschitz bound (positive quantity)
        # The learnable parameter is log(gamma), then we take gamma = exp(log_gamma)
        log_gamma = self.param("ln_gamma", init.constant(jnp.log(self.gamma)),
                               (1,), jnp.float32)
        if not self.trainable_lipschitz:
            _rng = jax.random.PRNGKey(0)
            log_gamma = init.constant(jnp.log(self.gamma))(_rng, (1,), jnp.float32)
            
        # Apply the Lipschitz bound
        sqrt_gamma = jnp.sqrt(jnp.exp(log_gamma))
        x = sqrt_gamma * inputs
        
        # Evaluate the network hidden layers
        for _, nz in enumerate(self.hidden_sizes):
            x = SandwichLayer(nz, 
                              activation=self.activation,
                              use_bias=self.use_bias,
                              kernel_init=self.kernel_init)(x)
        
        # Treat the output layer separately
        kinit = init.zeros_init() if self.init_output_zero else self.kernel_init
        x = SandwichLayer(self.output_size, 
                          is_output=True, 
                          use_bias=self.use_bias,
                          kernel_init=kinit)(x)
        x = sqrt_gamma * x
        
        return x
