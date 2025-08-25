import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import numpy as np 
from robustnn.solver.DYS import DavisYinSplit
from robustnn.plnet_torch.orthogonal import Params, cayley, norm, CayleyLinear

# this function should only be used for testing purpose
# it avoids the bacckpropogation of the parameters
# and initializes the tensor with [1, 2, ..., prod(shape)]
def seq_init_(tensor):
    """In-place init with [1,2,...], safe for nn.Parameter."""
    with torch.no_grad():
        numel = tensor.numel()
        values = torch.arange(1, numel + 1, dtype=tensor.dtype, device=tensor.device)
        tensor.view(-1).copy_(values)  # copy into tensor in place
    return tensor

def seq_init_transposed_(tensor):
    """In-place init with [1,2,...] but filled in column-major order (transposed)."""
    with torch.no_grad():
        numel = tensor.numel()
        values = torch.arange(1, numel + 1, dtype=tensor.dtype, device=tensor.device)
        # reshape in transposed form, then transpose back to match tensor.shape
        reshaped = values.view(tensor.shape[1], tensor.shape[0]).t()
        tensor.copy_(reshaped)
    return tensor


class MonLipLayer(nn.Module):
    def __init__(self, 
                 features: int, 
                 unit_features: Sequence[int],
                 mu: float = None,
                 nu: float = None,
                 tau: float = None,
                 is_mu_fixed: bool = False,
                 is_nu_fixed: bool = False,
                 is_tau_fixed: bool = False,
                 act: nn.Module = nn.ReLU()):
        super().__init__()
        self.is_mu_fixed = is_mu_fixed
        self.is_nu_fixed = is_nu_fixed
        self.is_tau_fixed = is_tau_fixed
        known = [mu is not None, nu is not None, tau is not None]
        if sum(known) < 2:
            raise ValueError("At least two of mu, nu, tau must be specified.")

        # Compute missing parameter
        if mu is None:
            mu = nu / tau
        elif nu is None:
            nu = mu * tau
        elif tau is None:
            tau = nu / mu

        # Register properly
        if is_mu_fixed:
            self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32))
        else:
            self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))

        if is_nu_fixed:
            self.nu = nn.Parameter(torch.tensor(nu, dtype=torch.float32))
        else:
            self.register_buffer("nu", torch.tensor(nu, dtype=torch.float32))

        if is_tau_fixed:
            self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        else:
            self.register_buffer("tau", torch.tensor(tau, dtype=torch.float32))

        # self.mu = mu
        # self.nu = nu  
        self.units = unit_features
        self.Fq = nn.Parameter(torch.empty(sum(self.units), features))
        nn.init.xavier_normal_(self.Fq)
        # seq_init_transposed_(self.Fq)
        self.fq = nn.Parameter(torch.empty((1,)))
        nn.init.constant_(self.fq, norm(self.Fq))
        self.by = nn.Parameter(torch.zeros(features))
        Fr, fr, b = [], [], []
        nz_1 = 0
        for nz in self.units:
            R = nn.Parameter(torch.empty((nz, nz+nz_1)))
            nn.init.xavier_normal_(R)
            # seq_init_transposed_(R)
            r = nn.Parameter(torch.empty((1,)))
            nn.init.constant_(r, norm(R))
            Fr.append(R)
            fr.append(r)
            b.append(nn.Parameter(torch.zeros(nz)))
            nz_1 = nz
        self.Fr = nn.ParameterList(Fr)
        self.fr = nn.ParameterList(fr)
        self.bs = nn.ParameterList(b)
        # cached weights
        self.Q = None 
        self.R = None 
        self.act = act

    def forward(self, x):
        sqrt_gam = math.sqrt(self.nu - self.mu)
        sqrt_2 = math.sqrt(2.)
        if self.training:
            self.Q, self.R = None, None 
            Q = cayley(self.fq * self.Fq / norm(self.Fq))
            R = [cayley(fr * Fr / norm(Fr)) for Fr, fr in zip(self.Fr, self.fr)]
        else:
            if self.Q is None:
                with torch.no_grad():
                    self.Q = cayley(self.fq * self.Fq / norm(self.Fq))
                    self.R = [cayley(fr * Fr / norm(Fr)) for Fr, fr in zip(self.Fr, self.fr)]
            Q, R = self.Q, self.R 

        xh = sqrt_gam * x @ Q.T
        yh = []
        hk_1 = xh[..., :0]
        idx = 0 
        for k, nz in enumerate(self.units):
            xk = xh[..., idx:idx+nz]
            gh = sqrt_2 * (self.act(sqrt_2 * torch.cat((xk, hk_1), dim=-1) @ R[k].T + self.bs[k]) ) @ R[k]
            # gh = sqrt_2 * F.relu (sqrt_2 * torch.cat((xk, hk_1), dim=-1) @ R[k].T + self.bs[k]) @ R[k]
            hk = gh[..., :nz] - xk
            gk = gh[..., nz:]
            yh.append(hk_1-gk)
            idx += nz 
            hk_1 = hk 
        yh.append(hk_1)

        yh = torch.cat(yh, dim=-1)
        y = 0.5 * ((self.mu + self.nu) * x + sqrt_gam * yh @ Q) + self.by 
        return y
    
    def _get_explicit_params(self):
        """
        Get explicit parameters for the MonLip layer.
        Returns:
            dict: Dictionary containing the explicit parameters.
        """
        gam = self.nu-self.mu
        by = self.by
        bs = self.bs
        # bh = torch.cat(bs, axis=0)
        bh = torch.cat([b for b in bs], dim=0)
        QT = cayley((self.fq / norm(self.Fq.T, eps=0)) * self.Fq.T)
        Q = QT.T
        sqrt_2g, sqrt_g2 = math.sqrt(2. * gam), math.sqrt(gam / 2.)

        V, S = [], []
        STks, BTks = [], []
        Ak_1s = [torch.zeros((0, 0)).numpy(force=True)]
        idx, nz_1 = 0, 0
        for k, nz in enumerate(self.units):
            Qk = Q[idx:idx+nz, :] 
            Fab = self.Fr[k].T
            fab = self.fr[k]
            ABT = cayley((fab / norm(Fab, eps=0)) * Fab)

            # todo: check the dimension here
            ATk, BTk = ABT[:nz, :], ABT[nz:, :]
            QTk_1, QTk = QT[:, idx-nz_1:idx], QT[:, idx:idx+nz]
            STk = QTk @ ATk - QTk_1 @ BTk

            # calculate V and S
            if k > 0:
                Ak, Bk = ATk.T, BTk.T
                V.append((2 * Bk @ ATk_1).numpy(force=True))
                S.append((Ak @ Qk - Bk @ Qk_1))
            else:
                Ak = ATk.T
                S.append(ABT.T @ Qk)
            ATk_1, Qk_1 = Ak.T, Qk
            
            STks.append(STk.numpy(force=True))
            BTks.append(BTk.numpy(force=True))
            Ak_1s.append(ATk.T.numpy(force=True))
            idx += nz
            nz_1 = nz

        Ak_1s=Ak_1s[:-1]
        S = torch.cat(S, axis=0).numpy(force=True)

        return Params(
            mu=self.mu.numpy(force=True),
            nu=self.nu.numpy(force=True),
            gam=self.nu.numpy(force=True) - self.mu.numpy(force=True),
            units=self.units,
            V=V,
            S=S,
            by=by.numpy(force=True),
            bh=bh.numpy(force=True),
            sqrt_2g=sqrt_2g,
            sqrt_g2=sqrt_g2,
            STks=STks,
            Ak_1s=Ak_1s,
            BTks=BTks,
            bs=[b.numpy(force=True) for b in bs],
        )

    
    def inverse(self, y,
                alpha: float = 1.0,
                inverse_activation_fn: callable = F.relu,
                iterations: int = 200,
                Lambda: float = 1.0):
        mon_params = self._get_explicit_params()

        # y to b
        # inverse of equation 12
        # bz = (y - e.by) / e.sqrt_2g
        bz = mon_params.sqrt_2g/mon_params.mu * (y-mon_params.by) @ mon_params.S.T + mon_params.bh
        uk = np.zeros_like(bz)

        # print(f"bz: {bz} and uk: {uk}")

        # iterate until converge for zk using DYS solver
        # todo: might change this for loop to jitable loop
        for i in range(iterations):
            # iterate until converge for zk using DYS solver
            zk, uk = DavisYinSplit(uk, bz, mon_params, 
                inverse_activation_fn=inverse_activation_fn, 
                Lambda=Lambda,
                alpha=alpha)
            
            # print(f"zk: {zk} and uk: {uk} at iteration {i}")


        # z to x
        x = (y - mon_params.by - mon_params.sqrt_g2 * zk @ mon_params.S) / mon_params.mu


        # check loss here
        # import jax
        # diff = jnp.linalg.norm(y - self.__call__(x), axis=-1)
        # jax.debug.print(f"MonLipNet inverse loss: {jnp.mean(diff)}")
        return x