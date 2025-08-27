'''
This file is a test file for the PLNet model implementation under pytorch.

Todo: The inverse of the Bi-lipschitz netowrk is not implemented yet.

Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (May 2024)
'''

import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import numpy as np 
from robustnn.plnet_torch.bilipnet import BiLipNet
from robustnn.plnet_torch.orthogonal import Params
import numpy as np

class PLNet(nn.Module):
    def __init__(self, 
                 BiLipBlock: nn.Module,
                 add_constant: bool = False,
                 optimal_point: torch.Tensor = None):
        super().__init__()
        self.bln = BiLipBlock 
        self.use_bias = add_constant
        if add_constant:
            self.bias = nn.Parameter(torch.zeros(1)) 

        self.optimal_point = optimal_point

    def forward(self, x):
        x = self.bln(x)

        if self.optimal_point is not None:
            x0 = self.bln(self.optimal_point)
        else:
            x0 = torch.zeros_like(x)
        y = 0.5 * ((x - x0) ** 2).mean(dim=-1)

        if self.use_bias:
            y += self.bias
        return y 
    
    def direct_to_explicit(self, x_optimal = None, act_mon = lambda x: np.maximum(0, x)) -> Params:
        """
        Convert the direct parameters to explicit parameters.

        Args:
            x_optimal: The optimal point for the quadratic potential. 
                       (None if no update on optimal point)
                       The dimension of x_optimal should be the same as the input size of the model 
                       or the size of 1
            act_mon: The activation function for the monotone layers. Default is ReLU in numpy.
        """
        # check if we have an optimal point - use the new one from input, if no flow back to the original one
        optimal_point = self.optimal_point.numpy(force=True)
        if x_optimal is not None:
            optimal_point = x_optimal
        
        if optimal_point is not None:
            def f_function(x: np.array, explicit: Params) -> np.array:
                # call the bilipnet with the optimal point
                # f = g(x) - g(x_optimal)
                g_x = self.bln.explicit_call(x, explicit, act_mon)
                g_x_optimal = self.bln.explicit_call(optimal_point, explicit, act_mon)
                
                # Calculate the quadratic potential
                return g_x - g_x_optimal
        else:
            def f_function(x: np.array, explicit: Params) -> np.array:
                # call the bilipnet with the optimal point
                # f = g(x)
                return self.bln.explicit_call(x, explicit, act_mon)
        
        # get the bilipnet properties
        lipmin, lipmax, distortion = self.bln.get_bounds()

        # convert the bilipnet to explicit
        explicit_params = Params(
            bilip_layer=self.bln.direct_to_explicit(),
            f_function=f_function,
            c=self.bias if self.use_bias else 0.,
            optimal_point=optimal_point,
            lipmin=lipmin,
            lipmax=lipmax,
            distortion=distortion
        )

        return explicit_params

    def explicit_call(self, x: np.array, explicit: Params) -> np.array:
        """
        Explicit call for the PLNet layer.

        Args:
            x: Input tensor.
            explicit: Explicit parameters for the BiLipNet layer.
            x_optimal: The optimal point for the quadratic potential. 
                        (None if no update on optimal point)
        """
        # Get the bilipnet output
        f = explicit.f_function(x, explicit.bilip_layer)

        # Calculate the quadratic potential
        y = 0.5 * np.mean(np.square(f), axis=-1) + explicit.c

        return y
    
    def _get_bounds(self):
        """Get the bounds for the BiLipNet layer."""

        lipmin, lipmax, tau = self.bln.get_bounds()
        return lipmin, lipmax, tau
    
if __name__ == "__main__":
    batch_size=5
    features = 32
    units = [32, 64, 128]
    mu=0.5
    nu=2.0
    bln = BiLipNet(features, units, mu, nu)
    nparams = np.sum([p.numel() for p in bln.parameters() if p.requires_grad])
    print(nparams)
    model = PLNet(bln)
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(nparams)
    x=torch.randn((batch_size, features))
    y=model(x)