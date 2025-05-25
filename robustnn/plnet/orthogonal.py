'''
Unitary layer using Cayley transform.
This layer applies a learned orthogonal (unitary) transformation to the input
using the Cayley map, preserving 2-norms in the transformation process.

Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (May 2024)
'''
import jax.numpy as jnp
from flax import linen as nn 
from flax.struct import dataclass
from robustnn.utils import cayley
from flax.typing import Array, PrecisionLike

@dataclass
class DirectOrthogonalParams:
    """Data class to keep track of implicit params for Orthogonal layer."""
    W: Array
    a: Array
    b: Array

@dataclass
class ExplicitOrthogonalParams:
    """Data class to keep track of explicit params for Orthogonal layer."""
    R: Array
    b: Array


class Unitary(nn.Module):
    """Unitary linear transformation layer using a Cayley transform.

    This layer applies a learned orthogonal (unitary) transformation to the input
    using the Cayley map, preserving 2-norms in the transformation process.

    Example usage::

        >>> layer = Unitary(input_size=4)
        >>> x = jnp.ones((1, 4))
        >>> params = layer.init(jax.random.key(0), x)
        >>> y = layer.apply(params, x)

    Attributes:
        input_size: Size of the input features.
        use_bias: Whether to include a learnable bias term (default: True).
    """

    input_size: int
    use_bias: bool = True 

    def setup(self):
        """Setup method for the Unitary layer."""
        
        m = self.input_size 

        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (m, self.input_size),
                       jnp.float32)
        a = self.param('a', 
                       nn.initializers.constant(jnp.linalg.norm(W)), 
                       (1,),
                       jnp.float32)
        
        if self.use_bias: 
            b = self.param('b', nn.initializers.zeros_init(), (m,), jnp.float32)
        else:
            b = 0.

        self.direct = DirectOrthogonalParams(W=W, a=a, b=b)

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        '''
        Call method for the Unitary layer.
        This method applies the Cayley transform to the input tensor `x` and
        returns the transformed tensor `z`.
        The transformation is defined as:
            z = x @ R^T
        where `R` is the Cayley matrix obtained from the learned parameters.
        The transformation is designed to be orthogonal, preserving the 2-norm of the input.
        The Cayley matrix is computed using the learned weight matrix `W` and a scaling factor `a`.
        The weight matrix `W` is initialized using the Glorot normal initializer,
        and the scaling factor `a` is initialized to the norm of `W`.
        If `use_bias` is set to True, a learnable bias term is added to the output.
        The parameters `W`, `a`, and `b` are learned during training.
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            z: Output tensor of shape (batch_size, output_dim).
        '''
        explict = self._direct_to_explicit()
        return self._explicit_call(x, explict)
    
    def _direct_to_explicit(self) -> ExplicitOrthogonalParams:
        """Convert implicit parameters to explicit parameters."""
        W = self.direct.W
        a = self.direct.a
        R = cayley((a / jnp.linalg.norm(W)) * W)
        b = self.direct.b 
        return ExplicitOrthogonalParams(R=R, b=b)
    
    def _explicit_call(self, x: jnp.array, e: ExplicitOrthogonalParams) -> Array:
        """Call method for the Unitary layer using explicit parameters.
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            e: ExplicitOrthogonalParams object containing explicit parameters.
        Returns:
            z: Output tensor of shape (batch_size, output_dim).
        """
        R = e.R
        b = e.b
        z = x @ R.T 
        if self.use_bias: 
            z += b
        return z
    
    def _explicit_inverse_call(self, y: jnp.array, e: ExplicitOrthogonalParams) -> Array:
        """
        Inverse call method for the Unitary layer using explicit parameters.
        Args:
            y: Output tensor of shape (batch_size, output_dim).
            e: ExplicitOrthogonalParams object containing explicit parameters.
        Returns:
            x: Input tensor of shape (batch_size, input_dim).
        """
        R = e.R
        b = e.b 
        if self.use_bias: 
            y -= b
        
        x = y @ R
        return x
    
    
    #################### Convenient Wrappers ####################
    def inverse_call(self, params: dict, y: Array, explicit: ExplicitOrthogonalParams):
        """Evaluate the inverse of the explicit model for an orthogonal layer.

        Args:
            params (dict): Flax model parameters dictionary.
            y (Array): model outputs.
            explicit (ExplicitOrthogonalParams): explicit params (inverse it the same as forward).

        Returns:
            Array: model inputs.
        """
        return self.apply(params, y, explicit, method="_explicit_inverse_call")

    def explicit_call(self, params: dict, x: Array, explicit: ExplicitOrthogonalParams):
        """Evaluate the explicit model for an orthogonal layer.

        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): model inputs.
            explicit (ExplicitLBDNParams): explicit params.

        Returns:
            Array: model outputs.
        """
        return self.apply(params, x, explicit, method="_explicit_call")
    
    def direct_to_explicit(self, params: dict)-> ExplicitOrthogonalParams:
        """Convert from direct orthogonal layer params to explicit form for eval.

        Args:
            params (dict): Flax model parameters dictionary.
            
        Returns:
            ExplicitOrthogonalParams: explicit orthogonal layer params.
        """
        return self.apply(params, method="_direct_to_explicit")
    
    # def get_params(self)-> ExplicitOrthogonalParams:
    #     """Get explicit parameters for the Unitary layer."""
    #     self.explict = self._direct_to_explicit()
    #     R = self.explict.R
    #     b = self.explict.b

    #     params = {
    #         'R': R,
    #         'b': b
    #     }
    #     return params