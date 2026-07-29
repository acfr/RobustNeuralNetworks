# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

'''
PyTorch dynamic unitary layer using Cayley transform.

Author: Yurui Zhang.
'''

import torch
import torch.nn as nn

from robustnn.orthogonal_torch import cayley


class DynUnitary(nn.Module):
    """Dynamic (stateful) orthogonal layer using a Cayley transform.

    A small recurrent linear system whose stacked state-space matrix is
    orthogonal (norm-preserving, so it does not change the bi-Lipschitz bounds):

        x1 = x @ A.T + u @ B.T + bx     (next state)
        y  = x @ C.T + u @ D.T + by     (output)

    with `G = [[A, B], [C, D]]` orthogonal. The system is exactly invertible
    *given the next state* `x1` (a non-causal inverse).

    Args:
        input_size: input/output feature size (nu == ny).
        state_size: dimension of the internal state.
        bias: whether to include learnable biases bx, by (default: True).
        dtype: parameter dtype (default: torch.float32).
    """
    def __init__(self, input_size: int, state_size: int, bias: bool = True,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        k = state_size + input_size
        self.X = nn.Parameter(torch.empty(k, k, dtype=dtype))
        nn.init.xavier_normal_(self.X)
        if bias:
            self.bx = nn.Parameter(torch.zeros(state_size, dtype=dtype))
            self.by = nn.Parameter(torch.zeros(input_size, dtype=dtype))
        else:
            self.register_buffer('bx', torch.zeros(state_size, dtype=dtype))
            self.register_buffer('by', torch.zeros(input_size, dtype=dtype))

    def _blocks(self):
        nx = self.state_size
        G = cayley(self.X)
        return G[:nx, :nx], G[:nx, nx:], G[nx:, :nx], G[nx:, nx:]

    def forward(self, state: torch.Tensor, u: torch.Tensor):
        """Forward pass: (state, inputs) -> (next_state, outputs)."""
        A, B, C, D = self._blocks()
        x1 = state @ A.T + u @ B.T + self.bx
        y = state @ C.T + u @ D.T + self.by
        return x1, y

    def inverse(self, next_state: torch.Tensor, y: torch.Tensor):
        """Non-causal inverse: recover (prev_state, inputs) from
        (next_state, outputs) using the orthogonality of G."""
        A, B, C, D = self._blocks()
        xb = next_state - self.bx
        yb = y - self.by
        x_prev = xb @ A + yb @ C
        u = xb @ B + yb @ D
        return x_prev, u

    def initialize_carry(self, batch_size: int, dtype: torch.dtype = torch.float32,
                         device=None) -> torch.Tensor:
        """Initialise the dynamic-orthogonal state (zeros)."""
        return torch.zeros(batch_size, self.state_size, dtype=dtype, device=device)
