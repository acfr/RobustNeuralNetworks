# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Minimal PyTorch port of Recurrent Equilibrium Networks (RENs), with a
bi-Lipschitz parameterisation (`BiLipschitzREN`).

This mirrors the JAX implementation in `robustnn/ren_base_jax.py` and
`robustnn/ren_jax.py`. RENs are recurrent neural networks that are internally
stable (contracting) and satisfy user-defined incremental IQC bounds. A
bi-Lipschitz REN is the square case whose input-output map `G` obeys

    mu * ||u1 - u2|| <= ||G(u1) - G(u2)|| <= nu * ||u1 - u2||.

RENs originally introduced in: [Recurrent Equilibrium Networks: Flexible Dynamic
Models With Guaranteed Stability and Robustness](https://ieeexplore.ieee.org/document/10179161).

Author: Yurui Zhang.
'''

import torch
import torch.nn as nn
from typing import Tuple


def tril_equilibrium_layer(activation, D11: torch.Tensor, b: torch.Tensor
                           ) -> torch.Tensor:
    """Solve `w = activation(D11 @ w + b)` for lower-triangular `D11`.

    Activation must be monotone with slope restricted to `[0, 1]`. Builds the
    solution column-by-column so the operation is differentiable (no in-place
    writes to a leaf tensor).
    """
    n = D11.shape[0]
    cols = []
    for i in range(n):
        if i == 0:
            pre = b[..., 0]
        else:
            wi = torch.stack(cols, dim=-1)          # (..., i)
            Di = D11[i, :i]                          # (i,)
            pre = wi @ Di + b[..., i]
        cols.append(activation(pre))
    return torch.stack(cols, dim=-1)


@torch.no_grad()
def solve_full_layer(activation, D11: torch.Tensor, b: torch.Tensor,
                     tol: float = 1e-9, alpha: float = 0.6,
                     max_iter: int = 200) -> torch.Tensor:
    """Solve `w = activation(D11 @ w + b)` for a full (non-triangular) `D11`
    using Douglas-Rachford operator splitting. Used by the inverse REN."""
    w = torch.zeros_like(b)
    uk = torch.zeros_like(b)
    I = torch.eye(D11.shape[0], dtype=b.dtype, device=b.device)
    M = I + alpha * (I - D11)
    for _ in range(max_iter):
        uh = 2 * w - uk
        zh = torch.linalg.solve(M, (uh + alpha * b).transpose(-2, -1)).transpose(-2, -1)
        uk = uk - w + zh
        w_new = activation(uk)
        if torch.linalg.norm(w - w_new) < tol:
            w = w_new
            break
        w = w_new
    return w


def _chol_upper(A: torch.Tensor) -> torch.Tensor:
    """Upper-triangular Cholesky factor `U` with `A = U.T @ U`."""
    L = torch.linalg.cholesky(A)
    return L.transpose(-2, -1)


class ExplicitRENParams:
    """Simple container for explicit REN params (torch tensors)."""
    def __init__(self, A, B1, B2, C1, C2, D11, D12, D21, D22, bx, bv, by):
        self.A, self.B1, self.B2 = A, B1, B2
        self.C1, self.C2 = C1, C2
        self.D11, self.D12, self.D21, self.D22 = D11, D12, D21, D22
        self.bx, self.bv, self.by = bx, bv, by


class RENBase(nn.Module):
    """Base class for Recurrent Equilibrium Networks (RENs) in PyTorch.

    Attributes:
        input_size: number of input features (nu).
        state_size: number of internal states (nx).
        features: number of hidden neurons (nv).
        output_size: number of output features (ny).
        activation: activation module (default: nn.ReLU()).
        abar: upper bound on the contraction rate, `0 < abar <= 1` (default: 1).
        eps: regularisation for positive-definite matrices.
    """
    def __init__(self,
                 input_size: int,
                 state_size: int,
                 features: int,
                 output_size: int,
                 activation: nn.Module = None,
                 abar: float = 1.0,
                 eps: float = float(torch.finfo(torch.float32).eps),
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.features = features
        self.output_size = output_size
        self.activation = activation if activation is not None else nn.ReLU()
        self.abar = abar
        self.eps = eps

        nu, nx, nv, ny = input_size, state_size, features, output_size
        d = min(nu, ny)

        def kernel(*shape):
            w = nn.Parameter(torch.empty(*shape, dtype=dtype))
            nn.init.xavier_normal_(w) if w.ndim >= 2 else nn.init.zeros_(w)
            return w

        # Free, trainable direct params (mirrors DirectRENParams in jax).
        self.B2 = kernel(nx, nu)
        self.D12 = kernel(nv, nu)
        self.X = kernel(2 * nx + nv, 2 * nx + nv)
        self.p = nn.Parameter(torch.full((1,), float(self.X.detach().norm()), dtype=dtype))
        self.Y1 = kernel(nx, nx)
        self.C2 = kernel(ny, nx)
        self.D21 = kernel(ny, nv)
        self.X3 = nn.Parameter(torch.eye(d, dtype=dtype))
        self.Y3 = nn.Parameter(torch.zeros(d, d, dtype=dtype))
        self.Z3 = nn.Parameter(torch.zeros(abs(ny - nu), d, dtype=dtype))
        self.bx = nn.Parameter(torch.zeros(nx, dtype=dtype))
        self.bv = nn.Parameter(torch.zeros(nv, dtype=dtype))
        self.by = nn.Parameter(torch.zeros(ny, dtype=dtype))

    #################### Parameterisation helpers ####################

    def _x_to_h_contracting(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Convert the REN `X` matrix to the contracting part of `H` using the
        polar parameterisation."""
        H = X.T @ X
        H = p ** 2 * H / (X.norm() ** 2)
        I = torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
        return H + self.eps * I

    def _hmatrix_to_explicit(self, H: torch.Tensor, D22: torch.Tensor
                             ) -> ExplicitRENParams:
        """Convert the REN `H` matrix to explicit form given the direct params."""
        nx, nv = self.state_size, self.features

        H11 = H[:nx, :nx]
        H22 = H[nx:(nx + nv), nx:(nx + nv)]
        H33 = H[(nx + nv):(2 * nx + nv), (nx + nv):(2 * nx + nv)]
        H21 = H[nx:(nx + nv), :nx]
        H31 = H[(nx + nv):(2 * nx + nv), :nx]
        H32 = H[(nx + nv):(2 * nx + nv), nx:(nx + nv)]

        P_imp = H33
        F = H31
        E = (H11 + P_imp / (self.abar ** 2) + self.Y1 - self.Y1.T) / 2

        B1_imp = H32
        C1_imp = -H21
        Lambda_inv = 2.0 / torch.diagonal(H22)
        D11_imp = -torch.tril(H22, diagonal=-1)

        A_e = torch.linalg.solve(E, F)
        B1_e = torch.linalg.solve(E, B1_imp)
        B2_e = torch.linalg.solve(E, self.B2)

        C1_e = Lambda_inv[:, None] * C1_imp
        D11_e = Lambda_inv[:, None] * D11_imp
        D12_e = Lambda_inv[:, None] * self.D12

        return ExplicitRENParams(A_e, B1_e, B2_e, C1_e, self.C2, D11_e,
                                 D12_e, self.D21, D22, self.bx, self.bv, self.by)

    #################### Forward / inverse evaluation ####################

    def _direct_to_explicit(self) -> ExplicitRENParams:
        raise NotImplementedError(
            "RENBase should not be used directly. Use a parameterisation such "
            "as `BiLipschitzREN`."
        )

    def _explicit_call(self, x: torch.Tensor, u: torch.Tensor,
                       e: ExplicitRENParams) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x @ e.C1.T + u @ e.D12.T + e.bv
        w = tril_equilibrium_layer(self.activation, e.D11, b)
        x1 = x @ e.A.T + w @ e.B1.T + u @ e.B2.T + e.bx
        y = x @ e.C2.T + w @ e.D21.T + u @ e.D22.T + e.by
        return x1, y

    def _explicit_inverse(self, e: ExplicitRENParams) -> ExplicitRENParams:
        """Construct the explicit params of the inverse REN. Requires `D22` to
        be square and invertible (true for bi-Lipschitz RENs)."""
        D22_inv = torch.linalg.inv(e.D22)
        B2_D = e.B2 @ D22_inv
        D12_D = e.D12 @ D22_inv

        A_inv = e.A - B2_D @ e.C2
        B1_inv = e.B1 - B2_D @ e.D21
        C1_inv = e.C1 - D12_D @ e.C2
        C2_inv = -D22_inv @ e.C2
        D11_inv = e.D11 - D12_D @ e.D21
        D21_inv = -D22_inv @ e.D21
        bx_inv = e.bx - B2_D @ e.by
        bv_inv = e.bv - D12_D @ e.by
        by_inv = -D22_inv @ e.by
        return ExplicitRENParams(A_inv, B1_inv, B2_D, C1_inv, C2_inv, D11_inv,
                                 D12_D, D21_inv, D22_inv, bx_inv, bv_inv, by_inv)

    def _explicit_inverse_call(self, x: torch.Tensor, u: torch.Tensor,
                               e: ExplicitRENParams
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the inverse REN. The inverse `D11` is generally full, so the
        iterative Douglas-Rachford solver is used for the equilibrium layer."""
        b = x @ e.C1.T + u @ e.D12.T + e.bv
        w_eq = solve_full_layer(self.activation, e.D11, b)
        v = w_eq @ e.D11.T + b
        w = self.activation(v)
        x1 = x @ e.A.T + w @ e.B1.T + u @ e.B2.T + e.bx
        y = x @ e.C2.T + w @ e.D21.T + u @ e.D22.T + e.by
        return x1, y

    def forward(self, state: torch.Tensor, inputs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the REN: (state, inputs) -> (next_state, output)."""
        e = self._direct_to_explicit()
        return self._explicit_call(state, inputs, e)

    #################### Convenient wrappers ####################

    def direct_to_explicit(self) -> ExplicitRENParams:
        """Convert direct params to explicit forward params."""
        return self._direct_to_explicit()

    def direct_to_explicit_inverse(self) -> ExplicitRENParams:
        """Convert direct params to explicit inverse params."""
        return self._explicit_inverse(self._direct_to_explicit())

    def explicit_call(self, x, u, e: ExplicitRENParams):
        """Evaluate the forward REN given explicit params."""
        return self._explicit_call(x, u, e)

    def inverse_call(self, x, u, e: ExplicitRENParams):
        """Evaluate the inverse REN given explicit (inverse) params."""
        return self._explicit_inverse_call(x, u, e)

    def inverse(self, state: torch.Tensor, outputs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recover inputs from outputs: (state, outputs) -> (next_state, inputs)."""
        e_inv = self.direct_to_explicit_inverse()
        return self._explicit_inverse_call(state, outputs, e_inv)

    def initialize_carry(self, batch_size: int,
                         dtype: torch.dtype = torch.float32,
                         device=None) -> torch.Tensor:
        """Initialise the REN state (zeros)."""
        return torch.zeros(batch_size, self.state_size, dtype=dtype, device=device)


class BiLipschitzREN(RENBase):
    """A bi-Lipschitz REN (square, `input_size == output_size`).

    The incremental IQC matrices are derived from the bounds:

        Q = -alpha2 * I,   S = I,   R = -alpha1 * I,
        alpha1 = 2*mu*nu / (mu + nu),   alpha2 = 2 / (mu + nu),

    where `mu` is the lower (inverse-Lipschitz) bound and `nu` the upper
    (Lipschitz) bound.

    Attributes:
        mu: lower (inverse-Lipschitz) bound, `0 < mu < nu`.
        nu: upper (Lipschitz) bound.
    """
    def __init__(self,
                 input_size: int,
                 state_size: int,
                 features: int,
                 mu: float = 1.0,
                 nu: float = 10.0,
                 activation: nn.Module = None,
                 abar: float = 1.0,
                 eps: float = float(torch.finfo(torch.float32).eps),
                 dtype: torch.dtype = torch.float32):
        super().__init__(input_size, state_size, features, input_size,
                         activation=activation, abar=abar, eps=eps, dtype=dtype)
        if nu <= mu:
            raise ValueError("Require `nu > mu` for a bi-Lipschitz REN.")
        self.mu = mu
        self.nu = nu
        self._dtype = dtype

    def _get_qsr(self):
        n = self.input_size
        I = torch.eye(n, dtype=self._dtype)
        alpha1 = 2.0 * (self.mu * self.nu) / (self.mu + self.nu)
        alpha2 = 2.0 / (self.mu + self.nu)
        Q = -alpha2 * I
        S = I.clone()
        R = -alpha1 * I
        return Q, S, R

    def _direct_to_explicit(self) -> ExplicitRENParams:
        nu, nx, ny = self.input_size, self.state_size, self.output_size
        Q, S, R = self._get_qsr()
        I_ny = torch.eye(ny, dtype=Q.dtype)
        I_nu = torch.eye(nu, dtype=Q.dtype)

        # Adjust for numerical conditioning (matches `_adjust_iqc_params`).
        Q = Q - self.eps * I_ny
        R = R + self.eps * I_nu

        R_temp = R - S @ torch.linalg.solve(Q, S.T)
        LQ = _chol_upper(-Q)
        LR = _chol_upper(R_temp)

        # Construct D22 (square case: ny == nu, Z3 is empty).
        M = self.X3.T @ self.X3 + self.Y3 - self.Y3.T + self.eps * I_ny
        N = torch.linalg.solve((I_ny + M).T, (I_ny - M).T).T
        D22 = torch.linalg.solve(-Q, S.T) + torch.linalg.solve(LQ, N) @ LR

        # Construct H (Eqn. 28 of Revay et al. (2023)).
        C2_imp = (D22.T @ Q + S) @ self.C2
        D21_imp = (D22.T @ Q + S) @ self.D21 - self.D12.T

        R1 = R + S @ D22 + D22.T @ S.T + D22.T @ Q @ D22
        mul_Q = torch.cat((self.C2, self.D21, torch.zeros((ny, nx), dtype=Q.dtype)), dim=1)
        mul_R = torch.cat((C2_imp, D21_imp, self.B2.T), dim=1)
        Gamma_Q = mul_Q.T @ Q @ mul_Q
        Gamma_R = mul_R.T @ torch.linalg.solve(R1, mul_R)

        H = self._x_to_h_contracting(self.X, self.p) + Gamma_R - Gamma_Q
        return self._hmatrix_to_explicit(H, D22)
