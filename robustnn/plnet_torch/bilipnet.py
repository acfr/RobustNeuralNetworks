import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from robustnn.plnet_torch.monlipnet import MonLipLayer, CayleyLinear
import numpy as np

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
                 depth: int = 1,
                 act: nn.Module = nn.ReLU()):
        super().__init__()
        self.depth = depth

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

        mu = mu ** (1./depth)
        nu = nu ** (1./depth)
        tau = nu / mu

        olayer = [CayleyLinear(features, features) for _ in range(depth+1)]
        self.orth_layers = nn.Sequential(*olayer)
        mlayer = [MonLipLayer(features, unit_features, mu, nu, tau,
                              is_mu_fixed, is_nu_fixed, is_tau_fixed, act) for _ in range(depth)]
        self.mon_layers = nn.Sequential(*mlayer)

    def forward(self, x):
        for k in range(self.depth):
            x = self.orth_layers[k](x)
            x = self.mon_layers[k](x)
        x = self.orth_layers[self.depth](x)
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
        x = y
        for k in range(self.depth, 0, -1):
            # x_old = x
            # print(x)
            x = self.orth_layers[k].inverse(x)
            # print(f"Orthogonal layer {k} inverse error: {np.linalg.norm(self.orth_layers[k](torch.from_numpy(x).to("cuda")).detach().cpu().numpy()
            #                                               - x_old)}")
            
            # x_old = x
            # print(x)
            x = self.mon_layers[k-1].inverse(
                x, 
                alpha=alphas[k-1],
                inverse_activation_fn=inverse_activation_fns[k-1],
                iterations=iterations[k-1],
                Lambda=Lambdas[k-1])
            x = np.array(x)

            # print(f"x shape: {x.shape}, x_old shape: {x_old.shape}")
            # print(f"Monlip layer {k-1} inverse error: {np.linalg.norm(self.mon_layers[k-1](torch.from_numpy(x).to("cuda")).detach().cpu().numpy() 
            #                                               - x_old)}")
        # print(x)
        # x_old = x
        x = self.orth_layers[0].inverse( x)
        # print(f"Orthogonal layer 0 inverse error: {np.linalg.norm(self.orth_layers[0](torch.from_numpy(x).to("cuda")).detach().cpu().numpy()
        #                                                   - x_old)}")
        # print(x)
        return x