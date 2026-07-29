# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
PyTorch composition of bi-Lipschitz RENs with norm-preserving unitary layers.

This is the PyTorch counterpart of `robustnn.ren_composition_jax.CompositionREN`.
It stacks several `BiLipschitzREN` blocks interleaved with `Unitary` (Cayley)
layers; because the unitary layers are norm-preserving, the overall map keeps the
bi-Lipschitz bounds `[mu, nu]`, split geometrically across the REN blocks.

Architecture:

    [input_orth | dyn_in] -> [REN -> orth] x num_layers -> output_orth [-> dyn_out]

The (optional) input/output layers can be made dynamic (`DynUnitary`) via
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

from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn

from robustnn.ren_torch import BiLipschitzREN
from robustnn.orthogonal_torch import Unitary, cayley, norm
from robustnn.dyn_orthogonal_torch import DynUnitary

# A carry is a dict: {"rens": [s_0, ...], "dyn_in": Tensor|None, "dyn_out": Tensor|None}.
Carry = Dict[str, Any]


def _unitary_q(layer: Unitary) -> torch.Tensor:
    """Compute the (differentiable, torch) orthogonal matrix of a `Unitary`."""
    return cayley(layer.alpha * layer.weight / norm(layer.weight))


def _unitary_forward(layer: Unitary, x: torch.Tensor) -> torch.Tensor:
    """Forward of a `Unitary` layer kept in torch (reuses the cached forward)."""
    return layer(x)


def _unitary_inverse(layer: Unitary, y: torch.Tensor) -> torch.Tensor:
    """Inverse of a `Unitary` layer kept in torch (the built-in `inverse`
    returns numpy)."""
    Q = _unitary_q(layer)
    b = layer.bias if layer.bias is not None else 0.0
    return (y - b) @ Q


class CompositionREN(nn.Module):
    """A bi-Lipschitz REN built by composing RENs with unitary layers (PyTorch).

    Example usage:

        >>> import torch
        >>> from robustnn.ren_composition_torch import CompositionREN
        >>> model = CompositionREN(2, 4, 8, num_layers=3, mu=0.5, nu=5.0)
        >>> carry = model.initialize_carry(5)
        >>> u = torch.ones(5, 2)
        >>> new_carry, y = model(carry, u)

    Args:
        input_size: input/output feature size (square map, nu == ny).
        state_size: number of internal states (nx) for each REN block.
        features: number of hidden neurons (nv) for each REN block.
        num_layers: number of REN blocks.
        mu: overall lower (inverse-Lipschitz) bound (default 1.0).
        nu: overall upper (Lipschitz) bound (default 10.0).
        activation: REN activation function (default: torch.relu).
        use_bias: whether unitary layers use a bias (default: True).
        dyn_orth_at_input: use a dynamic orthogonal layer at the input (default: False).
        dyn_orth_at_output: append a dynamic orthogonal layer at the output
            (default: False).
        dyn_state_multiplier: dynamic-orthogonal state size = this * state_size
            (default: 50).
        dtype: parameter dtype (default: torch.float32).
    """
    def __init__(self, input_size: int, state_size: int, features: int,
                 num_layers: int, mu: float = 1.0, nu: float = 10.0,
                 activation=torch.relu, use_bias: bool = True,
                 dyn_orth_at_input: bool = False, dyn_orth_at_output: bool = False,
                 dyn_state_multiplier: int = 50,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if num_layers < 1:
            raise ValueError("`num_layers` must be >= 1.")
        if nu <= mu:
            raise ValueError("Require `nu > mu` for a bi-Lipschitz network.")

        self.input_size = input_size
        self.state_size = state_size
        self.features = features
        self.num_layers = num_layers
        self.mu = mu
        self.nu = nu
        self.use_bias = use_bias
        self.dyn_orth_at_input = dyn_orth_at_input
        self.dyn_orth_at_output = dyn_orth_at_output
        self.dyn_state_multiplier = dyn_state_multiplier

        # Split the bounds geometrically across the REN blocks.
        layer_mu = mu ** (1.0 / num_layers)
        layer_nu = nu ** (1.0 / num_layers)

        dyn_size = dyn_state_multiplier * state_size

        # Input layer: static or dynamic orthogonal.
        if dyn_orth_at_input:
            self.dyn_in = DynUnitary(input_size, dyn_size, bias=use_bias, dtype=dtype)
            self.input_orth = None
        else:
            self.input_orth = Unitary(input_size, input_size, bias=use_bias)
            self.dyn_in = None

        self.rens = nn.ModuleList([
            BiLipschitzREN(input_size, state_size, features,
                           mu=layer_mu, nu=layer_nu, activation=activation,
                           dtype=dtype)
            for _ in range(num_layers)
        ])
        self.orths = nn.ModuleList([
            Unitary(input_size, input_size, bias=use_bias)
            for _ in range(num_layers)
        ])
        self.output_orth = Unitary(input_size, input_size, bias=use_bias)

        # Optional dynamic orthogonal layer appended after the output orth.
        if dyn_orth_at_output:
            self.dyn_out = DynUnitary(input_size, dyn_size, bias=use_bias, dtype=dtype)
        else:
            self.dyn_out = None

    def forward(self, carry: Carry, inputs: torch.Tensor
                ) -> Tuple[Carry, torch.Tensor]:
        """Forward pass: (carry, inputs) -> (next_carry, outputs)."""
        new_carry: Carry = {"rens": [None] * self.num_layers,
                            "dyn_in": None, "dyn_out": None}

        if self.dyn_orth_at_input:
            d_in, x = self.dyn_in(carry["dyn_in"], inputs)
            new_carry["dyn_in"] = d_in
        else:
            x = _unitary_forward(self.input_orth, inputs)

        for i in range(self.num_layers):
            s_i, x = self.rens[i](carry["rens"][i], x)
            new_carry["rens"][i] = s_i
            x = _unitary_forward(self.orths[i], x)

        x = _unitary_forward(self.output_orth, x)

        if self.dyn_orth_at_output:
            d_out, x = self.dyn_out(carry["dyn_out"], x)
            new_carry["dyn_out"] = d_out

        return new_carry, x

    def inverse(self, carry: Carry, outputs: torch.Tensor
                ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Causal inverse that *ignores* the dynamic-orthogonal layers.

        Inverts only the invertible part (output orth + RENs + static orths). For
        a network with an input `dyn_in`, the returned signal is the dyn-orth
        *output* (the input to the first static block), not the true network
        input. Requires `dyn_orth_at_output == False`.

        Returns:
            (recovered_ren_states, recovered_signal).
        """
        if self.dyn_orth_at_output:
            raise ValueError(
                "The causal 'ignore' inverse is not available when "
                "`dyn_orth_at_output=True`; use `inverse_noncausal` instead."
            )
        x = _unitary_inverse(self.output_orth, outputs)
        rec_states: List[torch.Tensor] = [None] * self.num_layers
        for i in range(self.num_layers - 1, -1, -1):
            x = _unitary_inverse(self.orths[i], x)
            s_i, x = self.rens[i].inverse(carry["rens"][i], x)
            rec_states[i] = s_i
        if not self.dyn_orth_at_input:
            # Static input orth is invertible; recover the true input.
            x = _unitary_inverse(self.input_orth, x)
        # A dynamic input orth (if any) is intentionally skipped.
        return rec_states, x

    def inverse_noncausal(self, carry: Carry, new_carry: Carry,
                          outputs: torch.Tensor
                          ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Non-causal inverse using the saved dynamic-orthogonal states.

        Uses `carry["rens"]` (the REN states *before* the forward step) for the
        REN inverses and `new_carry["dyn_in"]` / `new_carry["dyn_out"]` (the
        dynamic states *after* the forward step) for the dynamic-orthogonal
        inverses, exactly recovering the true network input.

        Returns:
            (recovered_ren_states, recovered_inputs).
        """
        x = outputs
        if self.dyn_orth_at_output:
            _, x = self.dyn_out.inverse(new_carry["dyn_out"], x)
        x = _unitary_inverse(self.output_orth, x)

        rec_states: List[torch.Tensor] = [None] * self.num_layers
        for i in range(self.num_layers - 1, -1, -1):
            x = _unitary_inverse(self.orths[i], x)
            s_i, x = self.rens[i].inverse(carry["rens"][i], x)
            rec_states[i] = s_i

        if self.dyn_orth_at_input:
            _, x = self.dyn_in.inverse(new_carry["dyn_in"], x)
        else:
            x = _unitary_inverse(self.input_orth, x)
        return rec_states, x

    def initialize_carry(self, batch_size: int,
                         dtype: torch.dtype = torch.float32,
                         device=None) -> Carry:
        """Initialise the internal states for every REN block (and dynamic
        orthogonal layers, if enabled)."""
        carry: Carry = {"rens": [], "dyn_in": None, "dyn_out": None}
        for ren in self.rens:
            carry["rens"].append(ren.initialize_carry(batch_size, dtype, device))
        dyn_size = self.dyn_state_multiplier * self.state_size
        if self.dyn_orth_at_input:
            carry["dyn_in"] = torch.zeros(batch_size, dyn_size, dtype=dtype,
                                          device=device)
        if self.dyn_orth_at_output:
            carry["dyn_out"] = torch.zeros(batch_size, dyn_size, dtype=dtype,
                                           device=device)
        return carry

    def get_bounds(self) -> Tuple[float, float]:
        """Get the (lower, upper) Lipschitz bounds of the composition."""
        return self.mu, self.nu
