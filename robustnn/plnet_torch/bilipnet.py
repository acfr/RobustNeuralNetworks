# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
BilipNet is a neural network architecture that combines Unitary and Monotone Lipschitz layers.
It is a subclass of the torch.nn Module and is designed to be used with PyTorch and numpy.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (Aug 2024)
'''

import torch.nn as nn
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
        """
        BiLipNet as described in the paper (Same as jax version).
        arguments:
            features: input and output feature size (same) 
            unit_features: list of hidden unit sizes for each monotone layer
            mu: lower Lipschitz bound (if None, will be computed)
            nu: upper Lipschitz bound (if None, will be computed)
            tau: Lipschitz constant (if None, will be computed)
            is_mu_fixed: whether mu is fixed during training
            is_nu_fixed: whether nu is fixed during training
            is_tau_fixed: whether tau is fixed during training
            depth: number of MonLip layers (and unitary layers = depth + 1)
            act: activation function in torch (default ReLU)
        """
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
        """
        Forward pass of the BiLipNet.
        arguments:
            x: (batch_size, features) in torch tensor
        return: 
            (batch_size, features) in torch tensor
        """
        for k in range(self.depth):
            x = self.orth_layers[k](x)
            x = self.mon_layers[k](x)
        x = self.orth_layers[self.depth](x)
        return x 
    
    def direct_to_explicit(self) -> Params:
        """
        Convert direct params to explicit params.
        return: 
            Params object containing explicit parameters of 
                Monlip layer and unitary/orthogonal layer (in numpy array)"""
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
        arguments:
            x (np.array): Input array of shape (batch_size, input_dim).
            explicit (Params): Params object containing explicit parameters.
            act_mon (callable): Activation function for the MonLip layers. (need to be numpy version!)
        return: 
            (np.array): Output numpy array of shape (batch_size, input_dim).
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
    
    def inverse(self, y: np.array,
                alphas: Sequence[float],
                inverse_activation_fns: Sequence[callable],
                iterations: Sequence[int],
                Lambdas: Sequence[float]):
        """        
        Inverse of the BiLipNet.
        arguments:
            y (numpy): Ouput array to be inverted.
            alphas (Sequence[float]): Sequence of alpha values for each layer.
            inverse_activation_fns (Sequence[callable]): Sequence of inverse activation functions for each layer.
            iterations (Sequence[int]): Number of iterations for each layer's solver.
            Lambdas (Sequence[float]): Step sizes for each layer's solver.
        returns:
            numpy array: Inverted Ouput.
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
        x = self.orth_layers[0].inverse(x)
        return x