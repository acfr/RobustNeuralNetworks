import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from robustnn.plnet_torch.monlipnet import MonLipLayer, CayleyLinear

class BiLipNet(nn.Module):
    def __init__(self,
                 features: int, 
                 unit_features: Sequence[int],
                 mu: float = None,
                 nu: float = None,
                 tau: float = None,
                 is_mu_fixed: bool = False,
                 is_nu_fixed: bool = False,
                 is_tau_fixed: bool = False,
                 nlayer: int = 1,
                 act: nn.Module = nn.ReLU):
        super().__init__()
        self.nlayer = nlayer

        # set up mu, nu, and tau
        # Determine which one to compute
        known = [mu is not None, nu is not None, tau is not None]
        if sum(known) < 2:
            raise ValueError("At least two of mu, nu, tau must be specified.")

        # Compute missing parameter
        if mu is None:
            mu = nu / tau
        elif nu is None:
            nu = mu * tau

        mu = mu ** (1./nlayer)
        nu = nu ** (1./nlayer)
        tau = nu / mu

        olayer = [CayleyLinear(features, features) for _ in range(nlayer+1)]
        self.orth_layers = nn.Sequential(*olayer)
        mlayer = [MonLipLayer(features, unit_features, mu, nu, tau,
                              is_mu_fixed, is_nu_fixed, is_tau_fixed, act) for _ in range(nlayer)]
        self.mon_layers = nn.Sequential(*mlayer)

    def forward(self, x):
        for k in range(self.nlayer):
            x = self.orth_layers[k](x)
            x = self.mon_layers[k](x)
        x = self.orth_layers[self.nlayer](x)
        return x 
    
    def inverse(self, y,
                alphas: Sequence[float],
                inverse_activation_fns: Sequence[callable],
                iterations: Sequence[int],
                Lambdas: Sequence[float]):

        """        Inverse of the BiLipNet.
        Args:
            y (torch.Tensor): Input tensor to be inverted.
            alphas (Sequence[float]): Sequence of alpha values for each layer.
            inverse_activation_fns (Sequence[callable]): Sequence of inverse activation functions for each layer.
            iterations (Sequence[int]): Number of iterations for each layer's solver.
            Lambdas (Sequence[float]): Step sizes for each layer's solver.
        Returns:
            torch.Tensor: Inverted tensor.
        """
        for k in range(self.depth, 0, -1):
            x = self.orth_layers[k].inverse(y)
            x = self.mon_layers[k-1].inverse(
                x, 
                alpha=alphas[k-1],
                inverse_activation_fn=inverse_activation_fns[k-1],
                iterations=iterations[k-1],
                Lambda=Lambdas[k-1])
        x = self.orth_layers[0].inverse( x)
        return x