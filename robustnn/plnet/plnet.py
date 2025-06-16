# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
PLNet is a neural network architecture that based on bilipnet.
It takes the quadratic potential of the output of the bilipnet.
This is useful for applications where we want to learn a potential function
that is Lipschitz continuous and has a known/unkown minimum.

Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (May 2024)
'''
import jax.numpy as jnp
from flax import linen as nn 
from flax.struct import dataclass
from robustnn.utils import cayley
from flax.typing import Array, PrecisionLike
from typing import Any, Sequence, Callable
from robustnn.plnet.bilipnet import BiLipNet, ExplicitBiLipParams, DirectBiLipParams

@dataclass
class DirectPLParams:
    """
    Data class to keep track of direct params for Bi-Lipschitz layer.
    """
    bilip_layer: DirectBiLipParams

    # c is the constant term in the quadratic potential
    c: Array = None

    
@dataclass
class ExplicitPLParams:
    """Data class to keep track of explicit params for Bi-Lipschitz layer."""
    bilip_layer: ExplicitBiLipParams

    # define the function inside the quadratic potential
    # f_function = g(x) if no optimal point is given
    # f_function = g(x) - g(x_optimal) if optimal point is given
    f_function: Callable

    # c is the constant term in the quadratic potential
    c: Array = None

    # some constant for model properties
    optimal_point: Array = None
    lipmin: float = 0.1
    lipmax: float = 10.0
    distortion: float = 100.0


class PLNet(nn.Module):
    """"
    PLNet is a neural network architecture that based on bilipnet.
    It takes the quadratic potential of the output of the bilipnet.
    This is useful for applications where we want to learn a potential function
    that is Lipschitz continuous and has a known/unkown minimum.
    The minimum can be provided initially, and the model will learn to
    approximate the potential function around that minimum. Also, the minimum 
    might be changed during runtime, and the model will adapt to that. 
    Moreover, it works if there is no minimum given.
    Example usage::
    
        >>> layer = PLNet(input_size=4, units=[4, 4])
        >>> x = jnp.ones((1, 4))
        >>> params = layer.init(jax.random.key(0), x)
        >>> y = layer.apply(params, x)

    Attributes:
        BiLipBlock: the base BiLipNet block (g)
        add_constant: Whether to add a learnable constant term to the quadratic (default: False).
        minimum: The known minimum/equilibrium point (default: None). This can be directly
            set to a value if known. We can encode this in the model and guarantee that 
            the minimum of PLNet is always at this point. 
    """
    BiLipBlock: nn.Module
    add_constant: float = False
    optimal_point: Array = None

    def setup(self):
        if self.add_constant:
            c = self.param('c', nn.initializers.constant(0.), (1,), jnp.float32)
        else:
            c = 0.

        self.direct = DirectPLParams(
            bilip_layer=self.BiLipBlock.direct,
            c=c,
        )


    def _direct_to_explicit(self, x_optimal = None) -> ExplicitPLParams:
        """
        Convert the direct parameters to explicit parameters.

        Args:
            x_optimal: The optimal point for the quadratic potential. 
                       (None if no update on optimal point)
                       The dimension of x_optimal should be the same as the input size of the model 
                       or the size of 1
        """
        # check if we have an optimal point - use the new one from input, if no flow back to the original one
        optimal_point = self.optimal_point
        if x_optimal is not None:
            optimal_point = x_optimal
        
        if optimal_point is not None:
            def f_function(x: jnp.array, explicit: ExplicitBiLipParams) -> jnp.array:
                # call the bilipnet with the optimal point
                # f = g(x) - g(x_optimal)
                g_x = self.BiLipBlock._explicit_call(x, explicit)
                g_x_optimal = self.BiLipBlock._explicit_call(optimal_point, explicit)
                
                # Calculate the quadratic potential
                return g_x - g_x_optimal
        else:
            def f_function(x: jnp.array, explicit: ExplicitBiLipParams) -> jnp.array:
                # call the bilipnet with the optimal point
                # f = g(x)
                return self.BiLipBlock._explicit_call(x, explicit)
        
        # get the bilipnet properties
        lipmin, lipmax, distortion = self.BiLipBlock._get_bounds()

        # convert the bilipnet to explicit
        explicit_params = ExplicitPLParams(
            bilip_layer=self.BiLipBlock._direct_to_explicit(),
            f_function=f_function,
            c=self.direct.c,
            optimal_point=self.optimal_point,
            lipmin=lipmin,
            lipmax=lipmax,
            distortion=distortion
        )


        return explicit_params

    def _explicit_call(self, x: jnp.array, explicit: ExplicitPLParams) -> jnp.array:
        """
        Explicit call for the PLNet layer.

        Args:
            x: Input tensor.
            explicit: Explicit parameters for the BiLipNet layer.
            x_optimal: The optimal point for the quadratic potential. 
                        (None if no update on optimal point)
        """
        # Get the bilipnet output
        f = explicit.f_function(x, explicit.bilip_layer)

        # Calculate the quadratic potential
        y = 0.5 * jnp.mean(jnp.square(f), axis=-1) + explicit.c

        return y
    
    @nn.compact
    def __call__(self, x: jnp.array, x_optimal: jnp.array = None) -> jnp.array:
        """
        Call method for the PLNet layer.

        Args:
            x: Input tensor.
            x_optimal: The optimal point for the quadratic potential. 
                       (None if no update on optimal point)
                       The dimension of x_optimal should be the same as the x 
                       or the size of 1

        Returns:
            y: Output tensor.
        """
        explicit = self._direct_to_explicit(x_optimal)
        return self._explicit_call(x, explicit)  

    def explicit_call(self, params: dict, x: Array, explicit: ExplicitPLParams):
        """
        Evaluate the explicit model for a PLNet layer.

        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): model inputs.
            explicit (ExplicitPLParams): explicit params.

        Returns:
            Array: model outputs.
        """
        return self.apply(params, x, explicit, method="_explicit_call")
    
    def direct_to_explicit(self, params: dict, x_optimal: jnp.array = None) -> ExplicitPLParams:
        """
        Convert from direct PLNet params to explicit form for eval.

        Args:
            params (dict): Flax model parameters dictionary.
            x_optimal: The optimal point for the quadratic potential. 
                       (None if no update on optimal point)

        Returns:
            ExplicitPLParams: explicit PLNet layer params.
        """
        return self.apply(params, x_optimal=x_optimal, method="_direct_to_explicit")
    

    
