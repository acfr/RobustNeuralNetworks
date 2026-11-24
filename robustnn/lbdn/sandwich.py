"""
Sandwich layers for Lipschitz Bounded Deep Networks (LBDNs), from Wang & Manchester
(ICML '23). See `lbdn.py` for the LBDN built from them.
"""

import jax.numpy as jnp

from typing import Optional

from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Dtype, Array, PrecisionLike

from robustnn.utils import l2_norm, cayley, cayley_b, dot_lax
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
    """Data class to keep track of explicit params for Sandwich layer.
    
    Note that psi_d from the original paper is multiplied against A_T and B
    in the code (see _explicit_call) so it does not need to be stored.
    """
    A_T: Array
    B: Array
    b: Array


@dataclass
class DirectLinearParams:
    """Data class to keep track of direct params for a Lipschitz linear layer."""
    XY: Array
    a: Array
    b: Array


@dataclass
class ExplicitLinearParams:
    """Data class to keep track of explicit params for a Lipschitz linear layer."""
    B: Array
    b: Array


# An LBDN is a stack of Sandwich layers followed by a Lipschitz linear layer,
# so its parameter sequences hold a mix of the two.
DirectLayerParams = DirectSandwichParams | DirectLinearParams
ExplicitLayerParams = ExplicitSandwichParams | ExplicitLinearParams


def _cayley_angle_init(theta: float, std: float) -> Initializer:
    """Initializer for a Sandwich weight `XY` whose Cayley blocks are diagonal.

    For `XY` of shape `(n_in + n_out, n_out)` this returns
    `XY = [0; -tan(theta/2) E] + std * N(0, 1)` with `E = eye(n_in, n_out)`.
    When `std = 0`, `Z = V^T V` is diagonal and the Cayley transform gives
    `A_T = cos(theta) I` and `B_T = sin(theta) E` on the first `min(n_in, n_out)`
    units. Any remaining units get `A_T = 1` and `B_T = 0`.

    Use it through `SandwichLayer.identity_init` or `SandwichLinear.identity_init`,
    which choose `theta` for a requested gain.

    Args:
        theta (float): angle splitting each unit between the `A_T` and `B_T` blocks.
        std (float): standard deviation of the Gaussian noise added to `XY`.

    Returns:
        Initializer: a function `(key, shape, dtype) -> XY`.
    """
    def init_func(key, shape, dtype) -> Array:
        n_out = shape[1]
        V = -jnp.tan(theta / 2) * jnp.eye(shape[0] - n_out, n_out, dtype=dtype)
        XY = jnp.concatenate([jnp.zeros((n_out, n_out), dtype), V])
        return XY + init.normal(std)(key, shape, dtype)
    return init_func


class SandwichLayerBase(nn.Module):
    """Base class for Sandwich network layers.

    This class has the parameters common to all Sandwich layers: the stacked
    weight matrix `XY` that is passed through the Cayley transform, its norm
    scaling `a`, and the bias `b`. Subclasses add any layer-specific
    parameters and define the direct-to-explicit map and the layer call.

    The layer interface has been written similarly to `linen.Dense`.

    Attributes:
        input_size: the number of input features.
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).

        kernel_init: initializer function for the weight matrix (default: lecun_normal()).
        bias_init: initializer function for the bias (default: zeros_init()).

        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see ``jax.lax.Precision``
            for details.
    """
    input_size: int
    features: int
    use_bias: bool = True

    kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()

    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None

    def setup(self):
        """Initialise direct params"""
        dtype = self.param_dtype

        XY = self.param("XY", self.kernel_init,
                        (self.input_size + self.features, self.features),
                        dtype)
        a = self.param("a", init.constant(l2_norm(XY)), (1,), dtype)
        b = self.param("b", self.bias_init, (self.features,), dtype)

        self.direct = self._build_direct(XY, a, b)

    def __call__(self, inputs: Array) -> Array:
        """Call a Sandwich layer.

        Args:
            inputs (Array): layer inputs.

        Returns:
            Array: layer outputs.
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(inputs, explicit)

    def _scale_weights(self, ps: DirectLayerParams) -> Array:
        """Scale `XY` by `a / ||XY||` ready for the Cayley transform.

        Args:
            ps (DirectLayerParams): direct layer params.

        Returns:
            Array: the scaled stacked weight matrix.
        """
        return ps.a / l2_norm(ps.XY) * ps.XY


    ############### Specify these for each layer type ###############

    def _build_direct(self, XY: Array, a: Array, b: Array) -> DirectLayerParams:
        """Build the direct param struct, adding any layer-specific params.

        This is called from within `setup()`, so subclasses may call
        `self.param(...)` here to register extra parameters.
        """
        raise NotImplementedError(
            "SandwichLayerBase layers should not be constructed directly. " +
            "Choose a layer type instead (eg: `SandwichLayer`)."
        )

    def _direct_to_explicit(self) -> ExplicitLayerParams:
        """Convert from direct layer params to explicit form for eval."""
        raise NotImplementedError(
            "SandwichLayerBase layers should not be called. " +
            "Choose a layer type instead (eg: `SandwichLayer`)."
        )

    def _explicit_call(self, u: Array, e: ExplicitLayerParams) -> Array:
        """Evaluate the explicit model for a layer."""
        raise NotImplementedError(
            "SandwichLayerBase layers should not be called. " +
            "Choose a layer type instead (eg: `SandwichLayer`)."
        )


class SandwichLayer(SandwichLayerBase):
    """The 1-Lipschtiz Sandwich layer from Wang & Manchester (ICML '23).

    Example usage::

        >>> from robustnn.lbdn import SandwichLayer
        >>> import jax, jax.numpy as jnp

        >>> layer = SandwichLayer(input_size=3, features=4)
        >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'XY': (7, 4), 'a': (1,), 'b': (4,), 'd': (4,)}}

    Attributes:
        activation: Activation function to use (default: relu).
        psi_init: initializer function for the activation scaling (default: zeros_init()).

    See docs for `SandwichLayerBase` for the remaining arguments.

    Note: Only monotone activations are supported: `identity`, `relu`, `tanh`, `sigmoid`.
    """
    activation: ActivationFn = nn.relu
    psi_init: Initializer = init.zeros_init()

    @staticmethod
    def identity_init(gain: float = 1.0, std: float = 0.0) -> dict:
        """Initializers that make the layer compute `sigma(gain * E^T u + b)`.

        `E = eye(input_size, features)` keeps the first `min(input_size, features)`
        inputs and pads any extra outputs with zeros. The Cayley blocks are set to
        `A_T = cos(t) I` and `B_T = sin(t) E` with `sin(2t) = gain`, and
        `psi = 1 / (sqrt(2) cos(t))`. The explicit weights are then
        `A_T psi = I / sqrt(2)`, the same output side as a unit-gain layer, and
        `B / psi = gain E^T / sqrt(2)`, so only the input matrix carries the gain.
        `gain = 1` gives `A_T = B_T = I / sqrt(2)` and `psi = 1`.

        With `std = 0`, the padded units are identical and stay that way under
        gradient descent. A small `std` gives them different starting points.

        Example usage::

            >>> layer = SandwichLayer(3, 8, **SandwichLayer.identity_init(0.5, std=1e-3))

        Args:
            gain (float): gain on the layer input, in `[0, 1]` (default: 1.0).
            std (float): std of Gaussian noise added to `XY` (default: 0.0, exact).

        Returns:
            dict: `kernel_init` and `psi_init` keyword arguments for the layer.
        """
        theta = jnp.arcsin(gain) / 2
        return dict(
            kernel_init=_cayley_angle_init(theta, std),
            psi_init=init.constant(-jnp.log(jnp.sqrt(2.0) * jnp.cos(theta))),
        )

    def _build_direct(self, XY: Array, a: Array, b: Array) -> DirectSandwichParams:
        """Add the activation scaling `d` to the shared direct params."""
        d = self.param("d", self.psi_init, (self.features,), self.param_dtype)
        return DirectSandwichParams(XY, a, d, b)

    def _direct_to_explicit(self) -> ExplicitSandwichParams:
        """Convert from direct Sandwich params to explicit form for eval.

        The activation scaling `psi` is folded into `A_T` and `B` here rather
        than applied on every call because it's loop-invariant.
        
        TODO: Doing this can lead to floating-point error accumulation in the grads.
        TODO: Revert it back to the original version if this becomes a problem.

        Returns:
            ExplicitSandwichParams: explicit Sandwich params.
        """
        ps = self.direct
        A_T, B_T = cayley(self._scale_weights(ps), return_split=True)

        # Clip d to avoid over/underflow. `diag(psi) @ B` scales the rows of B
        # and `A_T @ diag(psi)` the columns of A_T, so both are broadcasts.
        psi_d = jnp.exp(jnp.clip(ps.d, min=-20.0, max=20.0))
        return ExplicitSandwichParams(A_T * psi_d, B_T.T / psi_d[:, None], ps.b)

    def _explicit_call(self, u: Array, e: ExplicitSandwichParams) -> Array:
        """Evaluate the explicit model for a Sandwich layer.

        Args:
            u (Array): layer inputs.
            e (ExplicitSandwichParams): explicit params.

        Returns:
            Array: layer outputs.
        """
        sqrt2 = self.param_dtype(jnp.sqrt(2.0))
        x = sqrt2 * dot_lax(u, e.B)
        if self.use_bias:
            x += e.b
        return sqrt2 * dot_lax(self.activation(x), e.A_T)


class SandwichLinear(SandwichLayerBase):
    """A linear layer whose weight matrix satisfies `||B|| <= 1`.

    This is the output layer of an LBDN from Wang & Manchester (ICML '23).
    It shares the Cayley-parameterised weight of a `SandwichLayer`, but has
    no activation and hence no activation scaling `d`.

    Example usage::

        >>> from robustnn.lbdn import SandwichLinear
        >>> import jax, jax.numpy as jnp

        >>> layer = SandwichLinear(input_size=3, features=4)
        >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
        >>> jax.tree_util.tree_map(jnp.shape, params)
        {'params': {'XY': (7, 4), 'a': (1,), 'b': (4,)}}

    See docs for `SandwichLayerBase` for the full list of arguments.
    """

    @staticmethod
    def identity_init(gain: float = 1.0, std: float = 0.0) -> dict:
        """Initializers that make the weight a scaled identity, `B = gain * E^T`.

        `E = eye(input_size, features)` keeps the first `min(input_size, features)`
        inputs and pads any extra outputs with zeros. Returns a dict to match
        `SandwichLayer.identity_init`.

        Example usage::

            >>> layer = SandwichLinear(3, 3, **SandwichLinear.identity_init(0.5))

        Args:
            gain (float): gain in `[0, 1]` (default: 1.0).
            std (float): std of Gaussian noise added to `XY` (default: 0.0, exact).

        Returns:
            dict: the `kernel_init` keyword argument for the layer.
        """
        return dict(kernel_init=_cayley_angle_init(jnp.arcsin(gain), std))

    def _build_direct(self, XY: Array, a: Array, b: Array) -> DirectLinearParams:
        """The shared direct params are all this layer needs."""
        return DirectLinearParams(XY, a, b)

    def _direct_to_explicit(self) -> ExplicitLinearParams:
        """Convert from direct params to explicit form for eval.

        Only the `B` block of the Cayley transform is needed here, so we skip
        the linear solve for `A`.

        Returns:
            ExplicitLinearParams: explicit params.
        """
        ps = self.direct
        B_T = cayley_b(self._scale_weights(ps))
        return ExplicitLinearParams(B_T.T, ps.b)

    def _explicit_call(self, u: Array, e: ExplicitLinearParams) -> Array:
        """Evaluate the explicit model for a Lipschitz linear layer.

        Args:
            u (Array): layer inputs.
            e (ExplicitLinearParams): explicit params.

        Returns:
            Array: layer outputs.
        """
        x = dot_lax(u, e.B)
        return x + e.b if self.use_bias else x
