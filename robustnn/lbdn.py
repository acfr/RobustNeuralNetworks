'''
Implementation of Lipschitz Bounded Deep Networks (Linear) in JAX/FLAX

Adapted from Julia implentation: https://github.com/acfr/RobustNeuralNetworks.jl

Authors: Nic Barbara (Mar '24, Feb '25), Jack Naylor (Sep '23) from the ACFR.

These networks are compatible with other FLAX modules.
'''

import jax
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
        explicit = self._direct_to_explicit(self.direct)
        return self.explicit_call(inputs, explicit)
        
    def explicit_call(self, u: Array, e: ExplicitSandwichParams) -> Array:
        """Evaluate the explicit model for a Sandwich layer.

        Args:
            u (Array): layer inputs.
            e (ExplicitSandwichParams): explicit params.

        Returns:
            Array: layer outputs.
        """
        return self._explicit_call(
            u, e, self.activation, self.use_bias, self.is_output
        )
    
    @staticmethod
    def _explicit_call(
        u: Array, 
        e: ExplicitSandwichParams,
        activation: ActivationFn,
        use_bias: bool = True,
        is_output: bool = False,
    ) -> Array:
        """Static method for explicit call of the Sandwhich layer.

        Args:
            u (Array): layer inputs.
            e (ExplicitSandwichParams): explicit params.
            activation (ActivationFn): activation function.
            use_bias (bool, optional): whether to use bias vector. Defaults to True.
            is_output (bool, optional): is this just an output layer. Defaults to False.

        Returns:
            Array: layer outputs.
            
        The only reason this is separate from `SandwichLayer.explicit_call()` is
        so that the direct-to-explicit mapping and model call can be split up outside
        of the SandwichLayer (e.g., in `LBDN`.) If Flax allowed us to access parts of
        a model created in the `setup()` method this would not be needed...
        """
            
        # If just the output layer, return Bx + b (or just Bx if no bias)
        if is_output:
            x = dot_lax(u, e.B)
            return x + e.b if use_bias else x
                
        # Regular sandwich layer
        x = jnp.sqrt(2.0) * dot_lax(u, ((jnp.diag(1 / e.psi_d)) @ e.B))
        if use_bias: 
            x += e.b
        return jnp.sqrt(2.0) * dot_lax(activation(x), (e.A_T * e.psi_d.T))
    
    @staticmethod
    def _direct_to_explicit(ps:DirectSandwichParams) -> ExplicitSandwichParams:
        """Convert from direct Sandwich params to explicit form for eval.

        Args:
            ps (DirectSandwichParams): direct Sandwich params.

        Returns:
            ExplicitSandwichParams: explicit Sandwich params.
        """
        # Clip d to avoid over/underflow and return
        A_T, B_T = cayley(ps.a / l2_norm(ps.XY) * ps.XY, return_split=True)
        psi_d = jnp.exp(jnp.clip(ps.d, a_min=-20.0, a_max=20.0))
        return ExplicitSandwichParams(A_T, B_T.T, psi_d, ps.b)
    
    @staticmethod
    def params_to_explicit(ps: dict) -> ExplicitSandwichParams:
        """Convert from Flax params dict to explicit Sandwich params.

        Args:
            ps (dict): Flax params dict `{"params": {<model_params>}}`.

        Returns:
            ExplicitSandwichParams: explicit Sandwich params.
        """
        direct = DirectSandwichParams(
            XY = ps["params"]["XY"],
            a = ps["params"]["a"],
            d = ps["params"]["d"],
            b = ps["params"]["b"]
        )
        return SandwichLayer._direct_to_explicit(direct)


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
        {'params': {'SandwichLayer_0': {'XY': (13, 8), 'a': (1,), 'b': (8,), 'd': (8,)}, 'SandwichLayer_1': {'XY': (24, 16), 'a': (1,), 'b': (16,), 'd': (16,)}, 'SandwichLayer_2': {'XY': (18, 2), 'a': (1,), 'b': (2,)}, 'ln_gamma': (1,)}}
    
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
        in_layers = (self.input_size,) + self.hidden_sizes[:-1]
        layers = [
            SandwichLayer(
                input_size=in_layers[k],
                features=self.hidden_sizes[k], 
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init
            )
            for k in range(len(self.hidden_sizes))
        ]
        
        kinit = init.zeros_init() if self.init_output_zero else self.kernel_init
        layers.append(
            SandwichLayer(
                input_size=self.hidden_sizes[-1],
                features=self.output_size, 
                is_output=True, 
                use_bias=self.use_bias,
                kernel_init=kinit
            )
        )
        
        self.log_gamma = log_gamma
        self.layers = layers
        
    def __call__(self, inputs: Array) -> Array:
        """Call an LBDN model.

        Args:
            inputs (Array): model inputs.

        Returns:
            Array: model outputs.
        """
        direct = self._get_direct_params()
        explicit = self._direct_to_explicit(direct)
        return self.explicit_call(inputs, explicit)
    
    def _get_direct_params(self) -> DirectLBDNParams:
        """Get the direct params for an LBDN.

        Returns:
            DirectLBDNParams: direct LBDN params.
        """
        return DirectLBDNParams([s.direct for s in self.layers], self.log_gamma)

    def explicit_call(self, u: Array, explicit: ExplicitLBDNParams):
        """Evaluate the explicit model for an LBDN model.

        Args:
            u (Array): model inputs.
            e (ExplicitLBDNParams): explicit params.

        Returns:
            Array: model outputs.
        """
        return self._explicit_call(
            u, explicit, self.gamma, self.activation, self.use_bias
        )
        
    @staticmethod
    def _explicit_call(
        u: Array, 
        explicit: ExplicitLBDNParams,
        gamma: jnp.float32, # type: ignore
        activation: ActivationFn,
        use_bias: bool = True
    ) -> Array:
        """Static method for explicit call of an LBDN model

        Args:
            u (Array): layer inputs.
            explicit (ExplicitLBDNParams): explicit params.
            gamma (jnp.float32): Lipschitz bound to impose.
            activation (ActivationFn): activation function.
            use_bias (bool, optional): whether to use bias vector. Defaults to True.

        Returns:
            Array: model outputs.
        
        Like with the `SandwichLayer`, the only reason this is separate from `LBDN.
        explicit_call()` is so that the direct-to-explicit mapping and model call can be 
        split up outside of the LBDN model (e.g., in `ScalableREN`).
        """
        # log_gamma not stored in the params dict if it's not trainable
        if explicit.log_gamma is None:
            sqrt_gamma = jnp.sqrt(gamma)
        else:
            sqrt_gamma = jnp.sqrt(jnp.exp(explicit.log_gamma))
        x = sqrt_gamma * u
        
        # Evaluate the Sandwich layers
        for e in explicit.layers[:-1]:
            x = SandwichLayer._explicit_call(
                x, e, activation, use_bias, is_output=False
            )
        x = SandwichLayer._explicit_call(
            x, explicit.layers[-1], activation, use_bias, is_output=True
        )
        
        # Finish with the second part of the Lipschitz bound
        return sqrt_gamma * x

    @staticmethod
    def _direct_to_explicit(ps: DirectLBDNParams) -> ExplicitLBDNParams:
        """Convert from direct LBDN params to explicit form for eval.

        Args:
            ps (DirectLBDNParams): direct LBDN params.

        Returns:
            ExplicitLBDNParams: explicit LBDN params.
        """
        layer_explicit_params = [
            SandwichLayer._direct_to_explicit(ps_k) for ps_k in ps.layers
        ]
        return ExplicitLBDNParams(layer_explicit_params, ps.log_gamma)
    
    @staticmethod
    def params_to_explicit(ps: dict) -> ExplicitLBDNParams:
        """Convert from Flax params dict to explicit LBDN params.

        Args:
            ps (dict): Flax params dict `{"params": {<model_params>}}`.

        Returns:
            ExplicitLBDNParams: explicit LBDN params.
        """
        
        # Grab all the layer paramters
        layer_keys = [k for k in ps["params"].keys() if "layers_" in k]
        explicit = []
        for key in layer_keys:
            layer_ps = {"params": ps["params"][key]}
            explicit.append(SandwichLayer.params_to_explicit(layer_ps))
            
        # Check to see if there's a trainable Lipschitz bound
        if "log_gamma" in ps["params"].keys():
            log_gamma = ps["params"]["log_gamma"]
        else:
            log_gamma = None
        
        return ExplicitLBDNParams(explicit, log_gamma)
    