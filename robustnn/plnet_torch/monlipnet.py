# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Monotone Lipschitz neural network layer 

Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (Aug 2024)
'''

import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import numpy as np 
from robustnn.solver.DYS import DavisYinSplit
from robustnn.plnet_torch.orthogonal import Params, cayley, norm

class MonLipNet(nn.Module):
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
        """
        Monotone Lipschitz layer as described in the paper (Same as jax version).
        arguments:
            features: input and output feature size (same)
            unit_features: list of hidden unit sizes for each monotone layer
            mu: lower Lipschitz bound (if None, will be computed)
            nu: upper Lipschitz bound (if None, will be computed)
            tau: Lipschitz constant (if None, will be computed)
            is_mu_fixed: whether mu is fixed during training
            is_nu_fixed: whether nu is fixed during training
            is_tau_fixed: whether tau is fixed during training
            act: activation function in torch (default ReLU)
        """
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

        self.units = unit_features
        self.Fq = nn.Parameter(torch.empty(sum(self.units), features))
        nn.init.xavier_normal_(self.Fq)
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
        '''
        Forward pass of the MonLip layer.
        arguments:
            x: (batch_size, features) in torch tensor
        return: 
            (batch_size, features) in torch tensor
        '''
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
            hk = gh[..., :nz] - xk
            gk = gh[..., nz:]
            yh.append(hk_1-gk)
            idx += nz 
            hk_1 = hk 
        yh.append(hk_1)

        yh = torch.cat(yh, dim=-1)
        y = 0.5 * ((self.mu + self.nu) * x + sqrt_gam * yh @ Q) + self.by 
        return y
    
    def get_bounds(self):
        """Get the current bounds."""
        mu = self.mu.item()
        nu = self.nu.item()
        tau = self.tau.item()
        return mu, nu, tau
    
    def direct_to_explicit(self):
        """
        Get explicit parameters for the MonLip layer.
        returns:
            Params: Params containing the explicit parameters. 
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

    def explicit_call(self, x: np.array, explicit: Params, 
                         act = lambda x: np.maximum(0, x)) -> np.array:
        """
        Apply the explicit parameters to the input tensor.
        arguments:
            x: (batch_size, features) in numpy array
            explicit: Params containing the explicit parameters (in numpy array)
            act: activation function for the monotone layers (need to be numpy version!)
        return: 
            (batch_size, features) in numpy array 
        """
        # building equation 8 in paper [https://arxiv.org/html/2402.01344v2]
        # y = mu * x + by + sum(sqrt(g/2) * zk @ STk.T)
        y = explicit.mu * x + explicit.by
        zk = x[..., :0]
        for k, nz in enumerate(self.units):
            # zk = act(Vk @ zk + sqrt(2g) * x @ STk + bs)
            zk = act( 2 * (zk @ explicit.Ak_1s[k]) @ explicit.BTks[k] + explicit.sqrt_2g * x @ explicit.STks[k] + explicit.bs[k])
            y += explicit.sqrt_g2 * zk @ explicit.STks[k].T
        return y
    
    def inverse(self, y: np.array,
                alpha: float = 1.0,
                inverse_activation_fn: callable = lambda x: np.maximum(0, x),
                iterations: int = 200,
                Lambda: float = 1.0):
        """
        Inverse of the MonLip layer using Davis-Yin splitting method.
        arguments:
            y: (batch_size, features) in numpy array
            alpha: alpha value for the solver
            inverse_activation_fn: inverse activation function (need to be numpy version!)
            iterations: number of iterations for the solver
            Lambda: step size for the solver
        """
        
        mon_params = self.direct_to_explicit()

        # y to b
        # inverse of equation 12
        # bz = (y - e.by) / e.sqrt_2g
        bz = mon_params.sqrt_2g/mon_params.mu * (y-mon_params.by) @ mon_params.S.T + mon_params.bh
        uk = np.zeros_like(bz)

        # iterate until converge for zk using DYS solver
        for i in range(iterations):
            # iterate until converge for zk using DYS solver
            zk, uk = DavisYinSplit(uk, bz, mon_params, 
                inverse_activation_fn=inverse_activation_fn, 
                Lambda=Lambda,
                alpha=alpha)
            
        # z to x
        x = (y - mon_params.by - mon_params.sqrt_g2 * zk @ mon_params.S) / mon_params.mu
        return x