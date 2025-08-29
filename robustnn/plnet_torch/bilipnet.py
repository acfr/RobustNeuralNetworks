import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from robustnn.plnet_torch.monlipnet import MonLipLayer, CayleyLinear
import numpy as np
from robustnn.plnet_torch.orthogonal import Params

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

        if is_tau_fixed:
            tau = tau ** (1./depth)
        else:
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
    
    def direct_to_explicit(self) -> Params:
        """Convert direct params to explicit params."""
        monlip_explict_layers = [
            layer.direct_to_explicit() for layer in self.mon_layers
        ]
        unitary_explict_layers = [
            layer.direct_to_explicit() for layer in self.orth_layers
        ]

        lipmin, lipmax, tau = self.get_bounds()
        # get the bilipnet properties
        return Params(monlip_layers=monlip_explict_layers,
                                   unitary_layers=unitary_explict_layers,
                                   lipmin=lipmin,
                                   lipmax=lipmax,
                                   distortion=tau)
    
    def explicit_call(self, x: np.array, explicit: Params, act_mon = lambda x: np.maximum(0, x)) -> np.array:
        """Call method for the BiLipNet layer using explicit parameters.
        Args:
            x (np.array): Input array of shape (batch_size, input_dim).
            explicit (Params): Params object containing explicit parameters.
            act_mon (callable): Activation function for the MonLip layers. (need to be numpy version!)
        """
        for k in range(self.depth):
            x = self.orth_layers[k].explicit_call( x, explicit.unitary_layers[k])
            x = self.mon_layers[k].explicit_call( x, explicit.monlip_layers[k], act_mon)
        x = self.orth_layers[self.depth].explicit_call( x, explicit.unitary_layers[self.depth])
        return x
    
    def get_bounds(self):
        """Get the bounds for the BiLipNet layer."""

        lipmin, lipmax, tau = 1., 1., 1.
        for k in range(self.depth):
            mu, nu, ta = self.mon_layers[k].get_bounds()
            lipmin *= mu 
            lipmax *= nu 
            tau *= ta 
        return lipmin, lipmax, tau
    
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
            x = self.orth_layers[k].inverse(x)
            
            x = self.mon_layers[k-1].inverse(
                x, 
                alpha=alphas[k-1],
                inverse_activation_fn=inverse_activation_fns[k-1],
                iterations=iterations[k-1],
                Lambda=Lambdas[k-1])
        x = self.orth_layers[0].inverse( x)
        return x