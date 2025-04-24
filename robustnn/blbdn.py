'''
Implementation of Bi-Lipschitz Bounded Deep Networks (Linear) in JAX/FLAX

Adapted from Julia implentation: https://github.com/acfr/RobustNeuralNetworks.jl

Authors: Michael Somerfield (Mar '25), Jack Naylor (Sep '23) from the ACFR.

These networks are compatible with other FLAX modules.
'''

import jax.numpy as jnp

from typing import Sequence, Optional

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array, PrecisionLike

from robustnn import lbdn, unitary
from robustnn.utils import l2_norm, cayley, dot_lax
from robustnn.utils import ActivationFn, Initializer

@dataclass
class DirectBLBDNParams:
    """Data class to keep track of direct params for LBDN."""
    layers: Sequence[lbdn.ExplicitLBDNParams]
    log_gamma: Array


@dataclass
class ExplicitBLBDNParams:
    """Data class to keep track of explicit params for LBDN."""
    layers: Sequence[lbdn.ExplicitLBDNParams]
    log_gamma: Array


class BLBDN(nn.Module):
    """Bi-Lipschitz-Bounded Deep Network.

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
    tau: jnp.float32 = 1.0 # TODO Five this a proper name
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
        self.unitary, self.lipnet = [], []

        layer_tau = (self.tau) ** (1/self.depth)

        nu, ny = 5, 2
        layers = (8, 16)
        gamma = jnp.float32(10)

        for _ in range(self.depth):
            self.uni.append(unitary.Unitary())
            self.mon.append(lbdn.LBDN(nu, layers, ny, gamma=gamma))
        self.uni.append(unitary.Unitary())




        # dtype = self.param_dtype

        # # Set up trainable/constant Lipschitz bound (positive quantity)
        # # The learnable parameter is log(gamma), then we take gamma = exp(log_gamma)
        # log_gamma = dtype(jnp.log(self.gamma))
        # if self.trainable_lipschitz:
        #     log_gamma = self.param("ln_gamma", init.constant(log_gamma),(1,), dtype)

        # # Build a list of Sandwich layers, but treat the output seperately
        # layers = []
        # is_output = False
        # kernel_init = self.kernel_init
        # in_layers = (self.input_size,) + self.hidden_sizes
        # out_layers = self.hidden_sizes + (self.output_size,)

        # for k in range(len(in_layers)):

        #     if k == len(in_layers): # Output layer
        #         is_output = True
        #         if self.init_output_zero:
        #             kernel_init = init.zeros_init()

        #     layers.append(
        #         SandwichLayer(
        #             input_size=in_layers[k],
        #             features=out_layers[k],
        #             activation=self.activation,
        #             use_bias=self.use_bias,
        #             kernel_init=kernel_init,
        #             is_output=is_output,
        #             param_dtype=dtype
        #         )
        #     )

        # self.layers = layers
        # self.direct = DirectLBDNParams([s.direct for s in layers], log_gamma)

    def __call__(self, inputs: Array) -> Array:
        """Call an LBDN model.

        Args:
            inputs (Array): model inputs.

        Returns:
            Array: model outputs.
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(inputs, explicit)

    def _explicit_call(self, u: Array, explicit: ExplicitBLBDNParams):
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

    def _direct_to_explicit(self) -> ExplicitBLBDNParams:
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
        return ExplicitBLBDNParams(layer_explicit_params, ps.log_gamma)


    #################### Convenient Wrappers ####################

    def explicit_call(self, params: dict, u: Array, explicit: ExplicitBLBDNParams):
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
