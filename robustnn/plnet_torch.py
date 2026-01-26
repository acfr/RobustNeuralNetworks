# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
PLNet is a neural network architecture that based on bilipnet.
It takes the quadratic potential of the output of the bilipnet.
This is useful for applications where we want to learn a potential function
that is Lipschitz continuous and has a known/unkown minimum.

Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (Aug 2024)
'''
import torch
import torch.nn as nn
import numpy as np 
from robustnn.bilipnet_torch import BiLipNet
from robustnn.orthogonal_torch import Params
import numpy as np

class PLNet(nn.Module):
    def __init__(self, 
                 BiLipBlock: BiLipNet,
                 add_constant: bool = False,
                 optimal_point: torch.Tensor = None):
        """
        PLNet is a neural network architecture that based on bilipnet.
        It takes the quadratic potential of the output of the bilipnet.
        This is useful for applications where we want to learn a potential function
        that is Lipschitz continuous and has a known/unkown minimum.
        arguments:
            BiLipBlock: an instance of BiLipNet
            add_constant: whether to add a constant term to the quadratic potential
            optimal_point: the optimal point for the quadratic potential (None if no optimal point)
        """
        super().__init__()
        self.bln = BiLipBlock 
        self.use_bias = add_constant
        if add_constant:
            self.bias = nn.Parameter(torch.zeros(1)) 

        self.optimal_point = optimal_point

    def forward(self, x):
        """
        Forward pass of the PLNet layer.
        arguments: 
            x: (batch_size, in_features) in torch tensor
        return: 
            (batch_size,) in torch tensor
        """
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
        arguments:
            x_optimal: The optimal point for the quadratic potential. 
                       (None if no update on optimal point)
                       The dimension of x_optimal should be the same as the input size of the model 
                       or the size of 1
            act_mon: The activation function for the monotone layers. Default is ReLU in numpy.
        """
        # check if we have an optimal point - use the new one from input, if no flow back to the original one
        optimal_point = None
        if self.optimal_point is not None:
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
        augments:
            x: Input numpy array.
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
