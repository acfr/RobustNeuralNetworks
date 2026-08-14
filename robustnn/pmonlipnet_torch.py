"""Conditioned monotone-Lipschitz layer for parameter-dependent BiLip nets."""

import math
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from robustnn.orthogonal_torch import Params, cayley, norm


def _resolve_bounds(mu, nu, tau):
    """Resolve the three equivalent Lipschitz-bound parameterizations."""
    known = (mu is not None, nu is not None, tau is not None)
    if sum(known) < 2:
        raise ValueError("At least two of mu, nu, and tau must be specified.")
    values = {
        (False, True, True): lambda: (nu / tau, nu, tau),
        (True, False, True): lambda: (mu, mu * tau, tau),
        (True, True, False): lambda: (mu, nu, nu / mu),
        (True, True, True): lambda: (mu, nu, tau),
    }
    return values[known]()


class PMonLipNet(nn.Module):
    """MonLipNet whose hidden biases are supplied as a single tensor ``b``.

    ``b`` has ``sum(unit_features)`` features, with one contiguous slice for
    each hidden layer.  The output bias is learned independently of ``b``.
    """

    def __init__(
        self,
        features: int,
        unit_features: Sequence[int],
        mu: float = None,
        nu: float = None,
        tau: float = None,
        is_mu_fixed: bool = False,
        is_nu_fixed: bool = False,
        is_tau_fixed: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.is_mu_fixed = is_mu_fixed
        self.is_nu_fixed = is_nu_fixed
        self.is_tau_fixed = is_tau_fixed
        mu, nu, tau = _resolve_bounds(mu, nu, tau)

        for name, value, fixed in (
            ("mu", mu, is_mu_fixed),
            ("nu", nu, is_nu_fixed),
            ("tau", tau, is_tau_fixed),
        ):
            tensor = torch.tensor(value, dtype=torch.float32)
            if fixed:
                self.register_buffer(name, tensor)
            else:
                setattr(self, name, nn.Parameter(tensor))

        self.units = tuple(unit_features)
        self.Fq = nn.Parameter(torch.empty(sum(self.units), features))
        nn.init.xavier_normal_(self.Fq)
        self.fq = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.fq, norm(self.Fq))
        self.by = nn.Parameter(torch.zeros(features))

        Fr, fr = [], []
        previous_units = 0
        for units in self.units:
            matrix = nn.Parameter(torch.empty(units, units + previous_units))
            nn.init.xavier_normal_(matrix)
            scale = nn.Parameter(torch.empty(1))
            nn.init.constant_(scale, norm(matrix))
            Fr.append(matrix)
            fr.append(scale)
            previous_units = units
        self.Fr = nn.ParameterList(Fr)
        self.fr = nn.ParameterList(fr)
        self.act = act
        self.Q_cached = None
        self.R_cached = None

    def _orthogonal_factors(self):
        if self.training:
            self.Q_cached = self.R_cached = None
            return (
                cayley(self.fq * self.Fq / norm(self.Fq)),
                [cayley(scale * matrix / norm(matrix)) for matrix, scale in zip(self.Fr, self.fr)],
            )
        if self.Q_cached is None:
            with torch.no_grad():
                self.Q_cached = cayley(self.fq * self.Fq / norm(self.Fq))
                self.R_cached = [
                    cayley(scale * matrix / norm(matrix))
                    for matrix, scale in zip(self.Fr, self.fr)
                ]
        return self.Q_cached, self.R_cached

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if b.shape[-1] != sum(self.units):
            raise ValueError(f"b must have {sum(self.units)} features; got {b.shape[-1]}.")

        Q, R = self._orthogonal_factors()
        sqrt_gam = torch.sqrt(self.nu - self.mu)
        xh = sqrt_gam * x @ Q.T
        yh, hidden = [], xh[..., :0]
        index = 0
        for units, matrix in zip(self.units, R):
            next_index = index + units
            xk = xh[..., index:next_index]
            previous_hidden = hidden
            gh = math.sqrt(2.0) * (
                self.act(
                    math.sqrt(2.0) * torch.cat((xk, hidden), dim=-1) @ matrix.T
                    + b[..., index:next_index]
                )
                @ matrix
            )
            hidden, gk = gh[..., :units] - xk, gh[..., units:]
            yh.append(previous_hidden - gk)
            index = next_index
        yh.append(hidden)
        return 0.5 * ((self.mu + self.nu) * x + sqrt_gam * torch.cat(yh, dim=-1) @ Q) + self.by

    def get_bounds(self):
        """Return the current lower bound, upper bound, and distortion."""
        mu, nu = self.mu.item(), self.nu.item()
        return mu, nu, self.tau.item() if self.is_tau_fixed else nu / mu

    def direct_to_explicit(self) -> Params:
        """Convert the Cayley core to the package's explicit NumPy form."""
        gam = self.nu - self.mu
        QT = cayley((self.fq / norm(self.Fq.T, eps=0)) * self.Fq.T)
        Q = QT.T
        sqrt_2g, sqrt_g2 = torch.sqrt(2.0 * gam), torch.sqrt(gam / 2.0)

        V, S, STks, BTks = [], [], [], []
        Ak_1s = [torch.zeros((0, 0), device=Q.device, dtype=Q.dtype).numpy(force=True)]
        index, previous_units = 0, 0
        for layer, units in enumerate(self.units):
            Qk = Q[index:index + units, :]
            Fab = self.Fr[layer].T
            ABT = cayley((self.fr[layer] / norm(Fab, eps=0)) * Fab)
            ATk, BTk = ABT[:units, :], ABT[units:, :]
            QTk_1, QTk = QT[:, index - previous_units:index], QT[:, index:index + units]
            STk = QTk @ ATk - QTk_1 @ BTk
            if layer:
                Ak, Bk = ATk.T, BTk.T
                V.append((2.0 * Bk @ ATk_1).numpy(force=True))
                S.append(Ak @ Qk - Bk @ Qk_1)
            else:
                Ak = ATk.T
                S.append(ABT.T @ Qk)
            ATk_1, Qk_1 = Ak.T, Qk
            STks.append(STk.numpy(force=True))
            BTks.append(BTk.numpy(force=True))
            Ak_1s.append(ATk.T.numpy(force=True))
            index, previous_units = index + units, units

        return Params(
            mu=self.mu.numpy(force=True),
            nu=self.nu.numpy(force=True),
            gam=gam.numpy(force=True),
            units=self.units,
            V=V,
            S=torch.cat(S, dim=0).numpy(force=True),
            by=self.by.numpy(force=True),
            sqrt_2g=sqrt_2g.numpy(force=True),
            sqrt_g2=sqrt_g2.numpy(force=True),
            STks=STks,
            Ak_1s=Ak_1s[:-1],
            BTks=BTks,
        )

    def explicit_call(
        self,
        x: np.ndarray,
        b: np.ndarray,
        explicit: Params,
        act: Callable = lambda value: np.maximum(0, value),
    ) -> np.ndarray:
        """Evaluate the conditioned map using explicit NumPy parameters."""
        if b.shape[-1] != sum(self.units):
            raise ValueError(f"b must have {sum(self.units)} features; got {b.shape[-1]}.")
        y, z = explicit.mu * x + explicit.by, x[..., :0]
        index = 0
        for layer, units in enumerate(self.units):
            next_index = index + units
            z = act(
                2.0 * (z @ explicit.Ak_1s[layer]) @ explicit.BTks[layer]
                + explicit.sqrt_2g * x @ explicit.STks[layer]
                + b[..., index:next_index]
            )
            y += explicit.sqrt_g2 * z @ explicit.STks[layer].T
            index = next_index
        return y

    def inverse(
        self,
        y: np.ndarray,
        b: np.ndarray,
        alpha: float = 1.0,
        inverse_activation_fn: Callable = lambda value: np.maximum(0, value),
        iterations: int = 200,
        Lambda: float = 1.0,
    ) -> np.ndarray:
        """Invert the conditioned map for a fixed hidden-bias tensor."""
        from robustnn.solvers import DavisYinSplit

        params = self.direct_to_explicit()
        bz = params.sqrt_2g / params.mu * (y - params.by) @ params.S.T + b
        uk = np.zeros_like(bz)
        for _ in range(iterations):
            z, uk = DavisYinSplit(
                uk, bz, params, inverse_activation_fn=inverse_activation_fn,
                Lambda=Lambda, alpha=alpha,
            )
        return (y - params.by - params.sqrt_g2 * z @ params.S) / params.mu
