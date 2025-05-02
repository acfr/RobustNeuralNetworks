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

    optimal_point: Array = None
    
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
    lipmin: float
    lipmax: float
    distortion: float


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
    minimum: Array = None

    def setup(self):
        if self.add_constant:
            c = self.param('c', nn.initializers.constant(0.), (1,), jnp.float32)
        else:
            c = 0.

        self.direct = DirectPLParams(
            bilip_layers=self.BiLipBlock.direct,
            c=c,
            optimal_point=self.minimum
        )

    def _direct_to_explicit(self, x_optimal = None) -> ExplicitPLParams:
        """
        Convert the direct parameters to explicit parameters.

        Args:
            x_optimal: The optimal point for the quadratic potential. 
                       (None if no update on optimal point)
        """
        # update optimal
        if x_optimal is not None:
            self.direct.optimal_point = x_optimal
        
        if self.direct.optimal_point is not None:
            def f_function(x: jnp.array, explicit: ExplicitBiLipParams) -> jnp.array:
                # call the bilipnet with the optimal point
                # f = g(x) - g(x_optimal)
                g_x = self.BiLipBlock._explicit_call(x, explicit)
                g_x_optimal = self.BiLipBlock._explicit_call(self.direct.optimal_point, explicit)
                
                # Calculate the quadratic potential
                return g_x - g_x_optimal
        else:
            def f_function(x: jnp.array, explicit: ExplicitBiLipParams) -> jnp.array:
                # call the bilipnet with the optimal point
                # f = g(x)
                return self.BiLipBlock._explicit_call(x, explicit)
        
        # get the bilipnet properties
        lipmin = self.BiLipBlock.get_bounds()[0]
        lipmax = self.BiLipBlock.get_bounds()[1]
        distortion = self.BiLipBlock.get_bounds()[2]

        # convert the bilipnet to explicit
        explicit_params = ExplicitPLParams(
            bilip_layer=self.BiLipBlock._direct_to_explicit(),
            f_function=f_function,
            c=self.direct.c,
            optimal_point=self.direct.optimal_point,
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
    
    def get_bounds(self):
        """
        Get the bounds for the PLNet layer.

        Returns:
            lipmin: The minimum Lipschitz constant.
            lipmax: The maximum Lipschitz constant.
            tau: The distortion constant.
        """
        lipmin, lipmax, tau = self.BiLipBlock.get_bounds()
        return lipmin, lipmax, tau