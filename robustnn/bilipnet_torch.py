# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
BilipNet is a neural network architecture that combines Unitary and Monotone Lipschitz layers.
It is a subclass of the torch.nn Module and is designed to be used with PyTorch and numpy.
Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (Aug 2024)
'''

import torch.nn as nn
from typing import Callable, Optional, Sequence
from robustnn.monlipnet_torch import MonLipNet
import numpy as np
from robustnn.orthogonal_torch import Params, Unitary

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
        known = (mu is not None, nu is not None, tau is not None)
        if sum(known) < 2:
            raise ValueError("At least two of mu, nu, tau must be specified.")

        # Compute missing parameter using lookup table
        calc_map = {
            (False, True, True): lambda: (nu / tau, nu, tau),
            (True, False, True): lambda: (mu, mu * tau, tau),
            (True, True, False): lambda: (mu, nu, nu / mu),
            (True, True, True): lambda: (mu, nu, tau),
        }
        mu, nu, tau = calc_map[known]()

        # Apply per-layer scaling
        mu = mu ** (1./depth)
        nu = nu ** (1./depth)
        tau = tau ** (1./depth) if is_tau_fixed else nu / mu

        olayer = [Unitary(features, features) for _ in range(depth+1)]
        self.orth_layers = nn.Sequential(*olayer)
        mlayer = [MonLipNet(features, unit_features, mu, nu, tau,
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
    
    def _solver_arguments(self, alphas, inverse_activation_fns, iterations,
                          Lambdas, tolerances):
        values = {
            "alphas": [None] * self.depth if alphas is None else alphas,
            "inverse_activation_fns": (
                [lambda value: np.maximum(0, value)] * self.depth
                if inverse_activation_fns is None else inverse_activation_fns
            ),
            "iterations": [2000] * self.depth if iterations is None else iterations,
            "Lambdas": [1.0] * self.depth if Lambdas is None else Lambdas,
            "tolerances": [1e-6] * self.depth if tolerances is None else tolerances,
        }
        for name, sequence in values.items():
            if len(sequence) != self.depth:
                raise ValueError(f"{name} must contain {self.depth} values.")
        return values

    def inverse(self, y: np.array,
                alphas: Optional[Sequence[float]] = None,
                inverse_activation_fns: Optional[Sequence[Callable]] = None,
                iterations: Optional[Sequence[int]] = None,
                Lambdas: Optional[Sequence[float]] = None,
                tolerances: Optional[Sequence[float]] = None):
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
        return self.inverse_with_diagnostics(
            y, alphas, inverse_activation_fns, iterations, Lambdas, tolerances
        )[0]

    def inverse_with_diagnostics(self, y: np.array,
                alphas: Optional[Sequence[float]] = None,
                inverse_activation_fns: Optional[Sequence[Callable]] = None,
                iterations: Optional[Sequence[int]] = None,
                Lambdas: Optional[Sequence[float]] = None,
                tolerances: Optional[Sequence[float]] = None):
        """Invert the network and return per-block DYS diagnostics."""
        arguments = self._solver_arguments(
            alphas, inverse_activation_fns, iterations, Lambdas, tolerances
        )
        x = y
        residuals = [None] * self.depth
        steps = [None] * self.depth
        effective_alphas = [None] * self.depth
        for k in range(self.depth, 0, -1):
            x = self.orth_layers[k].inverse(x)
            x, residuals[k - 1], steps[k - 1], effective_alphas[k - 1] = (
                self.mon_layers[k - 1].inverse_with_diagnostics(
                    x,
                    alpha=arguments["alphas"][k - 1],
                    inverse_activation_fn=arguments["inverse_activation_fns"][k - 1],
                    iterations=arguments["iterations"][k - 1],
                    Lambda=arguments["Lambdas"][k - 1],
                    tolerance=arguments["tolerances"][k - 1],
                )
            )
        x = self.orth_layers[0].inverse(x)
        return (
            x,
            np.asarray(residuals),
            np.asarray(steps),
            np.asarray(effective_alphas),
        )
