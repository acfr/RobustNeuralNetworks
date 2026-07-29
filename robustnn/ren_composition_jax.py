# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Composition of bi-Lipschitz RENs with norm-preserving (orthogonal/unitary) layers.

This stacks several `BiLipschitzREN` blocks interleaved with `Unitary` (Cayley)
layers. Because the unitary layers are norm-preserving, the overall map keeps the
bi-Lipschitz bounds: the composition has lower bound `mu` and upper bound `nu`,
which are split geometrically across the `num_layers` REN blocks.

The architecture is:

    [input_orth | dyn_in] -> [REN -> orth] x num_layers -> output_orth [-> dyn_out]

where the (optional) input/output layers can be made dynamic (`DynUnitary`) via
`dyn_orth_at_input` / `dyn_orth_at_output`. Dynamic layers add memory but stay norm
preserving, so the bi-Lipschitz bounds are unchanged.

Two inverse modes are provided for networks containing dynamic-orthogonal layers:

    1. `inverse` (causal, "ignore dyn-orth"): inverts only the invertible part
       (static orths + RENs). The dynamic layers are skipped, so for an input
       `dyn_in` the result is the *signal at the dyn-orth output* (not the true
       network input). Requires no output dynamic layer.
    2. `inverse_noncausal`: uses the dynamic-orthogonal states saved during the
       forward pass (the *next* states) to invert the dynamic layers too, exactly
       recovering the true input.

Author: Yurui Zhang.
'''

import jax
import jax.numpy as jnp

from typing import Sequence, Tuple, List, Optional, Dict, Any
from flax import linen as nn
from flax.linen import initializers as init
from flax.struct import dataclass
from flax.typing import Array, Dtype

from robustnn.ren_jax import BiLipschitzREN
from robustnn.ren_base_jax import ExplicitRENParams
from robustnn.orthogonal_jax import Unitary, ExplicitOrthogonalParams
from robustnn.dyn_orthogonal_jax import DynUnitary, ExplicitDynOrthogonalParams
from robustnn.utils import ActivationFn, Initializer

# A carry is a dict: {"rens": [s_0, ...], "dyn_in": Array|None, "dyn_out": Array|None}.
Carry = Dict[str, Any]


@dataclass
class ExplicitCompRENParams:
    """Explicit params for a `CompositionREN` (forward direction)."""
    input_orth: Optional[ExplicitOrthogonalParams]
    rens: Sequence[ExplicitRENParams]
    orths: Sequence[ExplicitOrthogonalParams]
    output_orth: ExplicitOrthogonalParams
    dyn_in: Optional[ExplicitDynOrthogonalParams]
    dyn_out: Optional[ExplicitDynOrthogonalParams]

    # Model properties.
    lipmin: float
    lipmax: float


@dataclass
class ExplicitInverseCompRENParams:
    """Explicit params for a `CompositionREN` (inverse direction).
    
    The `rens` field holds the *inverse* explicit REN params; the orthogonal /
    dynamic-orthogonal fields hold their forward explicit params (the inverse of
    an orthogonal layer reuses the same matrices).
    """
    input_orth: Optional[ExplicitOrthogonalParams]
    rens: Sequence[ExplicitRENParams]
    orths: Sequence[ExplicitOrthogonalParams]
    output_orth: ExplicitOrthogonalParams
    dyn_in: Optional[ExplicitDynOrthogonalParams]
    dyn_out: Optional[ExplicitDynOrthogonalParams]

    # Model properties.
    lipmin: float
    lipmax: float


class CompositionREN(nn.Module):
    """A bi-Lipschitz REN built by composing RENs with unitary layers.

    Example usage:

        >>> import jax, jax.numpy as jnp
        >>> from robustnn.ren_composition_jax import CompositionREN

        >>> rng = jax.random.key(0)
        >>> k1, k2 = jax.random.split(rng)

        >>> io, nx, nv = 2, 4, 8
        >>> model = CompositionREN(io, nx, nv, num_layers=3, mu=0.5, nu=5.0)

        >>> batches = 5
        >>> carry = model.initialize_carry(k1, (batches, io))
        >>> inputs = jnp.ones((batches, io))
        >>> params = model.init(k2, carry, inputs)
        >>> new_carry, y = model.apply(params, carry, inputs)

    Attributes:
        input_size: input/output feature size (square map, nu == ny).
        state_size: number of internal states (nx) for each REN block.
        features: number of hidden neurons (nv) for each REN block.
        num_layers: number of REN blocks.
        mu: overall lower (inverse-Lipschitz) bound (default 1.0).
        nu: overall upper (Lipschitz) bound (default 10.0).
        activation: REN activation function (default: relu).
        use_bias: whether unitary layers use a bias (default: True).
        init_method: REN init method, "random" or "long_memory" (default: "random").
        dyn_orth_at_input: use a dynamic orthogonal layer at the input (default: False).
        dyn_orth_at_output: append a dynamic orthogonal layer at the output
            (default: False).
        dyn_state_multiplier: dynamic-orthogonal state size = this * state_size
            (default: 50).
        carry_init: initializer for REN state vectors (default: zeros).
        param_dtype: dtype for parameters (default: float32).
    """
    input_size: int
    state_size: int
    features: int
    num_layers: int
    mu: float = 1.0
    nu: float = 10.0
    activation: ActivationFn = nn.relu
    use_bias: bool = True
    init_method: str = "random"
    dyn_orth_at_input: bool = False
    dyn_orth_at_output: bool = False
    dyn_state_multiplier: int = 50
    carry_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32

    def setup(self):
        if self.num_layers < 1:
            raise ValueError("`num_layers` must be >= 1.")
        if self.nu <= self.mu:
            raise ValueError("Require `nu > mu` for a bi-Lipschitz network.")

        # Split the bounds geometrically across the REN blocks. The unitary
        # layers (static or dynamic) are norm-preserving so they do not affect
        # the bounds.
        layer_mu = self.mu ** (1.0 / self.num_layers)
        layer_nu = self.nu ** (1.0 / self.num_layers)

        dyn_size = self.dyn_state_multiplier * self.state_size

        # Input layer: static or dynamic orthogonal.
        if self.dyn_orth_at_input:
            self.dyn_in = DynUnitary(input_size=self.input_size,
                                     state_size=dyn_size,
                                     use_bias=self.use_bias,
                                     param_dtype=self.param_dtype)
            self.input_orth = None
        else:
            self.input_orth = Unitary(input_size=self.input_size,
                                      use_bias=self.use_bias)
            self.dyn_in = None

        self.rens = [
            BiLipschitzREN(self.input_size, self.state_size, self.features,
                           self.input_size, mu=layer_mu, nu=layer_nu,
                           activation=self.activation,
                           init_method=self.init_method,
                           carry_init=self.carry_init,
                           param_dtype=self.param_dtype)
            for _ in range(self.num_layers)
        ]
        self.orths = [
            Unitary(input_size=self.input_size, use_bias=self.use_bias)
            for _ in range(self.num_layers)
        ]
        self.output_orth = Unitary(input_size=self.input_size,
                                   use_bias=self.use_bias)

        # Optional dynamic orthogonal layer appended after the output orth.
        if self.dyn_orth_at_output:
            self.dyn_out = DynUnitary(input_size=self.input_size,
                                      state_size=dyn_size,
                                      use_bias=self.use_bias,
                                      param_dtype=self.param_dtype)
        else:
            self.dyn_out = None

    def __call__(self, carry: Carry, inputs: Array) -> Tuple[Carry, Array]:
        """Forward pass of the composition REN.

        Args:
            carry: dict with keys "rens" (list of per-REN states), "dyn_in" and
                "dyn_out" (dynamic-orthogonal states or None). See
                `initialize_carry`.
            inputs: model inputs, shape (batches, input_size).

        Returns:
            Tuple[Carry, Array]: (next_carry, outputs).
        """
        explicit = self._direct_to_explicit()
        return self._explicit_call(carry, inputs, explicit)

    #################### Direct -> explicit ####################

    def _direct_to_explicit(self) -> ExplicitCompRENParams:
        """Convert direct params to explicit params (forward direction)."""
        return ExplicitCompRENParams(
            input_orth=(None if self.dyn_orth_at_input
                        else self.input_orth._direct_to_explicit()),
            rens=[r._direct_to_explicit() for r in self.rens],
            orths=[o._direct_to_explicit() for o in self.orths],
            output_orth=self.output_orth._direct_to_explicit(),
            dyn_in=(self.dyn_in._direct_to_explicit() if self.dyn_orth_at_input else None),
            dyn_out=(self.dyn_out._direct_to_explicit()
                     if self.dyn_orth_at_output else None),
            lipmin=self.mu,
            lipmax=self.nu,
        )

    def _direct_to_explicit_inverse(self) -> ExplicitInverseCompRENParams:
        """Convert direct params to explicit params (inverse direction)."""
        return ExplicitInverseCompRENParams(
            input_orth=(None if self.dyn_orth_at_input
                        else self.input_orth._direct_to_explicit()),
            rens=[r._explicit_inverse(r._direct_to_explicit()) for r in self.rens],
            orths=[o._direct_to_explicit() for o in self.orths],
            output_orth=self.output_orth._direct_to_explicit(),
            dyn_in=(self.dyn_in._direct_to_explicit() if self.dyn_orth_at_input else None),
            dyn_out=(self.dyn_out._direct_to_explicit()
                     if self.dyn_orth_at_output else None),
            lipmin=self.mu,
            lipmax=self.nu,
        )

    #################### Forward / inverse evaluation ####################

    def _explicit_call(self, carry: Carry, inputs: Array,
                       e: ExplicitCompRENParams) -> Tuple[Carry, Array]:
        """Evaluate the forward composition using explicit params."""
        new_carry: Carry = {"rens": [None] * self.num_layers,
                            "dyn_in": None, "dyn_out": None}

        if self.dyn_orth_at_input:
            d_in, x = self.dyn_in._explicit_call(carry["dyn_in"], inputs, e.dyn_in)
            new_carry["dyn_in"] = d_in
        else:
            x = self.input_orth._explicit_call(inputs, e.input_orth)

        for i in range(self.num_layers):
            s_i, x = self.rens[i]._explicit_call(carry["rens"][i], x, e.rens[i])
            new_carry["rens"][i] = s_i
            x = self.orths[i]._explicit_call(x, e.orths[i])

        x = self.output_orth._explicit_call(x, e.output_orth)

        if self.dyn_orth_at_output:
            d_out, x = self.dyn_out._explicit_call(carry["dyn_out"], x, e.dyn_out)
            new_carry["dyn_out"] = d_out

        return new_carry, x

    def _explicit_inverse_call(self, carry: Carry, outputs: Array,
                               e: ExplicitInverseCompRENParams
                               ) -> Tuple[List[Array], Array]:
        """Causal inverse that *ignores* the dynamic-orthogonal layers.

        Inverts only the invertible part (output orth + RENs + static orths). For
        a network with an input `dyn_in`, the returned signal is the dyn-orth
        *output* (the input to the first static block), not the true network
        input. Requires `dyn_orth_at_output == False`.
        """
        if self.dyn_orth_at_output:
            raise ValueError(
                "The causal 'ignore' inverse is not available when "
                "`dyn_orth_at_output=True`; use `inverse_noncausal` instead."
            )
        x = self.output_orth._explicit_inverse_call(outputs, e.output_orth)
        rec_states: List[Array] = [None] * self.num_layers
        for i in range(self.num_layers - 1, -1, -1):
            x = self.orths[i]._explicit_inverse_call(x, e.orths[i])
            s_i, x = self.rens[i]._explicit_inverse_call(carry["rens"][i], x, e.rens[i])
            rec_states[i] = s_i
        if not self.dyn_orth_at_input:
            # Static input orth is invertible; recover the true input.
            x = self.input_orth._explicit_inverse_call(x, e.input_orth)
        # A dynamic input orth (if any) is intentionally skipped.
        return rec_states, x

    def _explicit_inverse_call_noncausal(self, carry: Carry, new_carry: Carry,
                                         outputs: Array,
                                         e: ExplicitInverseCompRENParams
                                         ) -> Tuple[List[Array], Array]:
        """Non-causal inverse using the saved dynamic-orthogonal states.

        Uses `carry["rens"]` (the REN states *before* the forward step) for the
        REN inverses and `new_carry["dyn_in"]` / `new_carry["dyn_out"]` (the
        dynamic states *after* the forward step) for the dynamic-orthogonal
        inverses, exactly recovering the true network input.
        """
        x = outputs
        if self.dyn_orth_at_output:
            _, x = self.dyn_out._explicit_inverse_call(new_carry["dyn_out"], x, e.dyn_out)
        x = self.output_orth._explicit_inverse_call(x, e.output_orth)

        rec_states: List[Array] = [None] * self.num_layers
        for i in range(self.num_layers - 1, -1, -1):
            x = self.orths[i]._explicit_inverse_call(x, e.orths[i])
            s_i, x = self.rens[i]._explicit_inverse_call(carry["rens"][i], x, e.rens[i])
            rec_states[i] = s_i

        if self.dyn_orth_at_input:
            _, x = self.dyn_in._explicit_inverse_call(new_carry["dyn_in"], x, e.dyn_in)
        else:
            x = self.input_orth._explicit_inverse_call(x, e.input_orth)
        return rec_states, x

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, input_shape: Tuple[int, ...]
                         ) -> Carry:
        """Initialise the internal states for every REN block (and dynamic
        orthogonal layers, if enabled).

        Returns:
            Carry: dict with "rens" (list of per-REN states) and "dyn_in" /
                "dyn_out" (dynamic-orthogonal states or None).
        """
        batch_dims = input_shape[:-1]
        ren_shape = batch_dims + (self.state_size,)
        dyn_shape = batch_dims + (self.dyn_state_multiplier * self.state_size,)

        carry: Carry = {"rens": [], "dyn_in": None, "dyn_out": None}
        for _ in range(self.num_layers):
            rng, sub = jax.random.split(rng)
            carry["rens"].append(self.carry_init(sub, ren_shape, self.param_dtype))
        if self.dyn_orth_at_input:
            carry["dyn_in"] = jnp.zeros(dyn_shape, self.param_dtype)
        if self.dyn_orth_at_output:
            carry["dyn_out"] = jnp.zeros(dyn_shape, self.param_dtype)
        return carry

    def _get_bounds(self) -> Tuple[float, float]:
        """Get the (lower, upper) Lipschitz bounds of the composition."""
        return self.mu, self.nu

    #################### Convenient Wrappers ####################

    def direct_to_explicit(self, params: dict) -> ExplicitCompRENParams:
        """Convert from direct params to explicit forward params."""
        return self.apply(params, method="_direct_to_explicit")

    def direct_to_explicit_inverse(self, params: dict
                                   ) -> ExplicitInverseCompRENParams:
        """Convert from direct params to explicit inverse params."""
        return self.apply(params, method="_direct_to_explicit_inverse")

    def explicit_call(self, params: dict, carry: Carry, inputs: Array,
                      explicit: ExplicitCompRENParams) -> Tuple[Carry, Array]:
        """Evaluate the forward composition given explicit params."""
        return self.apply(params, carry, inputs, explicit, method="_explicit_call")

    def inverse_call(self, params: dict, carry: Carry, outputs: Array,
                     explicit: ExplicitInverseCompRENParams
                     ) -> Tuple[List[Array], Array]:
        """Causal inverse (ignores dynamic-orthogonal layers) given explicit params."""
        return self.apply(params, carry, outputs, explicit,
                          method="_explicit_inverse_call")

    def inverse_call_noncausal(self, params: dict, carry: Carry, new_carry: Carry,
                               outputs: Array,
                               explicit: ExplicitInverseCompRENParams
                               ) -> Tuple[List[Array], Array]:
        """Non-causal inverse (uses saved dynamic states) given explicit params."""
        return self.apply(params, carry, new_carry, outputs, explicit,
                          method="_explicit_inverse_call_noncausal")

    def get_bounds(self, params: dict = None) -> Tuple[float, float]:
        """Get the (lower, upper) Lipschitz bounds of the composition."""
        return self._get_bounds()
