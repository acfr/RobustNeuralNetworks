'''
Implementation of Lipschitz Bounded Deep Networks (Linear) in JAX/FLAX

Adapted from Julia implentation: https://github.com/acfr/RobustNeuralNetworks.jl

Authors: Nic Barbara (Mar '24, Feb '25), Jack Naylor (Sep '23) from the ACFR.

These networks are compatible with other FLAX modules.
'''

import jax.numpy as jnp

from typing import Sequence, Optional

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array, PrecisionLike

from robustnn.utils import l2_norm, cayley, dot_lax
from robustnn.utils import ActivationFn, Initializer


@dataclass
class DirectSandwichParams:
    """Data class to keep track of direct params for Sandwich layer."""
    XY: Array
    a: Array
    d: Array
    b: Array


@dataclass
class ExplicitSandwichParams:
    """Data class to keep track of explicit params for Sandwich layer."""
    A_T: Array
    B: Array
    psi_d: Array
    b: Array


@dataclass
class DirectLBDNParams:
    """Data class to keep track of direct params for LBDN."""
    layers: Sequence[DirectSandwichParams]
    log_gamma: Array


@dataclass
class ExplicitLBDNParams:
    """Data class to keep track of explicit params for LBDN."""
    layers: Sequence[ExplicitSandwichParams]
    log_gamma: Array


class SandwichLayer(nn.Module):
    """The 1-Lipschtiz Sandwich layer from Wang & Manchester (ICML '23).
    
    The layer interface has been written similarly to `linen.Dense`.    

    Example usage::

        >>> from robustnn.networks.lbdn import SandwichLayer
        >>> import jax, jax.numpy as jnp

        >>> layer = SandwichLayer(input_size=3, features=4)
        >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
        >>> jax.tree_map(jnp.shape, params)
        {'params': {'XY': (7, 4), 'a': (1,), 'b': (4,), 'd': (4,)}}

    Attributes:
        input_size: the number of input features.
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).
        is_output: treat this as the output layer of an LBDN (default: False).
        activation: Activation function to use (default: relu).
        
        kernel_init: initializer function for the weight matrix (default: lecun_normal()).
        bias_init: initializer function for the bias (default: zeros_init()).
        psi_init: initializer function for the activation scaling (default: zeros_init()).
        
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see ``jax.lax.Precision``
            for details.
    """
    input_size: int
    features: int
    use_bias: bool = True
    is_output: bool = False
    activation: ActivationFn = nn.relu
    
    kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()
    psi_init: Initializer = init.zeros_init()
    
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    
    def setup(self):
        """Initialise direct Sandwich params."""
        dtype = self.param_dtype
        
        XY = self.param("XY", self.kernel_init, 
                        (self.input_size + self.features, self.features), 
                        dtype)
        a = self.param('a', init.constant(l2_norm(XY)), (1,), self.param_dtype)
        d = self.param('d', self.psi_init, (self.features,), self.param_dtype)
        b = self.param('b', self.bias_init, (self.features,), self.param_dtype)
        
        self.direct = DirectSandwichParams(XY, a, d, b)
    
    def __call__(self, inputs: Array) -> Array:
        """Call a Sandwich layer.

        Args:
            inputs (Array): layer inputs.

        Returns:
            Array: layer outputs.
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(inputs, explicit)
        
    def _explicit_call(self, u: Array, e: ExplicitSandwichParams) -> Array:
        """Evaluate the explicit model for a Sandwich layer.

        Args:
            u (Array): layer inputs.
            e (ExplicitSandwichParams): explicit params.

        Returns:
            Array: layer outputs.
        """
        if self.is_output:
            x = dot_lax(u, e.B)
            return x + e.b if self.use_bias else x
                
        x = jnp.sqrt(2.0) * dot_lax(u, ((jnp.diag(1 / e.psi_d)) @ e.B))
        if self.use_bias: 
            x += e.b
        return jnp.sqrt(2.0) * dot_lax(self.activation(x), (e.A_T * e.psi_d.T))
    
    def _direct_to_explicit(self) -> ExplicitSandwichParams:
        """Convert from direct Sandwich params to explicit form for eval.

        Args:
            ps (DirectSandwichParams): direct Sandwich params.

        Returns:
            ExplicitSandwichParams: explicit Sandwich params.
        """
        # Clip d to avoid over/underflow and return
        ps = self.direct
        A_T, B_T = cayley(ps.a / l2_norm(ps.XY) * ps.XY, return_split=True)
        psi_d = jnp.exp(jnp.clip(ps.d, a_min=-20.0, a_max=20.0))
        return ExplicitSandwichParams(A_T, B_T.T, psi_d, ps.b)


class LBDN(nn.Module):
    """Lipschitz-Bounded Deep Network.
    
    Example usage::
    
        >>> from robustnn.lbdn import LBDN
        >>> import jax, jax.numpy as jnp
        
        >>> nu, ny = 5, 2
        >>> layers = (8, 16)
        >>> gamma = jnp.float32(10)
        
        >>> model = LBDN(nu, layers, ny, gamma=gamma)
        >>> params = model.init(jax.random.key(0), jnp.ones((6,nu)))
        >>> jax.tree_map(jnp.shape, params)
        {'params': {'SandwichLayer_0': {'XY': (13, 8), 'a': (1,), 'b': (8,), 'd': (8,)}, 
        'SandwichLayer_1': {'XY': (24, 16), 'a': (1,), 'b': (16,), 'd': (16,)}, 
        'SandwichLayer_2': {'XY': (18, 2), 'a': (1,), 'b': (2,)}, 'ln_gamma': (1,)}}
    
    Attributes:
        input_size: the number of input features.
        hidden_sizes: Sequence of hidden layer sizes.
        output_size: the number of output features.
        gamma: upper bound on the Lipschitz constant (default: 1.0).
        activation: activation function to use (default: relu).
        
        kernel_init: initializer function for the weight matrix (default: lecun_normal()).
        bias_init: initializer function for the bias (default: zeros_init()).
        psi_init: initializer function for the activation scaling (default: zeros_init()).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        
        use_bias:whether to add a bias to the output (default: True).
        trainable_lipschitz: make the Lipschitz constant trainable (default: False).
        init_output_zero: initialize the network so its output is zero (default: False).
    
    Note: Only monotone activations are supported: `identity`, `relu`, `tanh`, `sigmoid`.
    """
    input_size: int
    hidden_sizes: Sequence[int]
    output_size: int
    gamma: jnp.float32 = 1.0 # type: ignore
    activation: ActivationFn = nn.relu
    
    kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()
    psi_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    
    use_bias: bool = True
    trainable_lipschitz: bool = False
    init_output_zero: bool = False
    
    def setup(self):
        """Initialise direct LBDN params.
        
        The setup is currently written in a rather convoluted way to make it possible
        to split up the direct-to-explicit transform and explicit model call. This is
        because anything initialised in `setup()` can't be accessed outside of `model.init()
        ` and `model.apply()` in Flax. Very frustrating.
        """
        
        dtype = self.param_dtype
        
        # Set up trainable/constant Lipschitz bound (positive quantity)
        # The learnable parameter is log(gamma), then we take gamma = exp(log_gamma)
        log_gamma = jnp.log(self.gamma)
        if self.trainable_lipschitz:
            log_gamma = self.param("ln_gamma", init.constant(log_gamma),(1,), dtype)
        
        # Build a list of Sandwich layers, but treat the output seperately
        layers = []
        is_output = False
        kernel_init = self.kernel_init
        in_layers = (self.input_size,) + self.hidden_sizes
        out_layers = self.hidden_sizes + (self.output_size,)
        
        for k in range(len(in_layers)):
            
            if k == len(in_layers): # Output layer
                is_output = True
                if self.init_output_zero:
                    kernel_init = init.zeros_init()
            
            layers.append(
                SandwichLayer(
                    input_size=in_layers[k],
                    features=out_layers[k], 
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_init=kernel_init,
                    is_output=is_output
                )
            )
        
        self.layers = layers
        self.direct = DirectLBDNParams([s.direct for s in layers], log_gamma)
        
    def __call__(self, inputs: Array) -> Array:
        """Call an LBDN model.

        Args:
            inputs (Array): model inputs.

        Returns:
            Array: model outputs.
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(inputs, explicit)
    
    def _explicit_call(self, u: Array, explicit: ExplicitLBDNParams):
        """Evaluate the explicit model for an LBDN model.

        Args:
            u (Array): model inputs.
            e (ExplicitLBDNParams): explicit params.

        Returns:
            Array: model outputs.
        """
        sqrt_gamma = jnp.sqrt(jnp.exp(explicit.log_gamma))
        x = sqrt_gamma * u
        
        for k, layer in enumerate(self.layers):
            x = layer._explicit_call(x, explicit.layers[k])
        
        return sqrt_gamma * x
    
    def _direct_to_explicit(self) -> ExplicitLBDNParams:
        """Convert from direct LBDN params to explicit form for eval.

        Args:
            None
            
        Returns:
            ExplicitLBDNParams: explicit LBDN params.
        """
        ps = self.direct
        layer_explicit_params = [
            layer._direct_to_explicit() for layer in self.layers
        ]
        return ExplicitLBDNParams(layer_explicit_params, ps.log_gamma)
    
    
    #################### Convenient Wrappers ####################
    
    def explicit_call(self, params: dict, u: Array, explicit: ExplicitLBDNParams):
        """Evaluate the explicit model for an LBDN model.

        Args:
            params (dict): Flax model parameters dictionary.
            u (Array): model inputs.
            e (ExplicitLBDNParams): explicit params.

        Returns:
            Array: model outputs.
        """
        return self.apply(params, u, explicit, method="_explicit_call")
    
    def direct_to_explicit(self, params: dict):
        """Convert from direct LBDN params to explicit form for eval.

        Args:
            params (dict): Flax model parameters dictionary.
            
        Returns:
            ExplicitLBDNParams: explicit LBDN params.
        """
        return self.apply(params, method="_direct_to_explicit")
