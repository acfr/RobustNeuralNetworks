# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Unitary layer using Cayley transform.
This layer applies a learned orthogonal (unitary) transformation to the input
using the Cayley map, preserving 2-norms in the transformation process.

Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (Aug 2024)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Params(nn.Module):
    """ Data class to keep track of explicit params for Unitary/Monotone layer."""
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v)  # not learnable
            else:
                setattr(self, k, v)

def cayley(W: torch.Tensor) -> torch.Tensor:
    """
    Cayley transform to obtain an orthogonal matrix from a skew-symmetric matrix.
    Args:
        W (torch.Tensor): Input tensor of shape (out_features, in_features).
    Returns:
        torch.Tensor: Orthogonal matrix of shape (out_features, in_features).
    """
    cout, cin = W.shape
    if cin > cout:
        return cayley(W.T).T
    U, V = W[:cin, :], W[cin:, :]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)
    A = U - U.T + V.T @ V
    iIpA = torch.inverse(I + A)

    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=0)

def norm(x, eps=0.0):
    """
    Compute the Frobenius norm of a tensor with numerical stability.
    Args:
        x (torch.Tensor): Input tensor.
        eps (float, optional): Small value to ensure numerical stability. Defaults to 0.0.
    Returns:
        torch.Tensor: Frobenius norm of the input tensor.
    """
    return x.norm() + eps

class Unitary(nn.Linear):
    """ Unitary layer using Cayley transform.
    This layer applies a learned orthogonal (unitary) transformation to the input
    using the Cayley map, preserving 2-norms in the transformation process.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize the Unitary layer.
        augments: in_features to handle cases where in_features > out_features.
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        """
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.empty(1).fill_(
            norm(self.weight).item()), requires_grad=True)

        self.Q_cached = None

    def reset_parameters(self):
        std = 1 / self.weight.shape[1] ** 0.5
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

        self.Q_cached = None

    def forward(self, X):
        """
        Forward pass of the Unitary layer.
        arguments: 
            x: (batch_size, in_features) in torch tensor
        return: 
            (batch_size, out_features) in torch tensor"""
        if self.training:
            self.Q_cached = None
            Q = cayley(self.alpha * self.weight / norm(self.weight))
        else:
            if self.Q_cached is None:
                with torch.no_grad():
                    self.Q_cached = cayley(
                        self.alpha * self.weight / norm(self.weight))
            Q = self.Q_cached

        return F.linear(X, Q, self.bias)
    
    def explicit_call(self, x: np.array, explicit: Params) -> np.array:
        """
        Forward method using explicit parameters.
        arguments: 
            x (np.array): Input array of shape (batch_size, input_dim).
            explicit (Params): Params object containing explicit parameters.
        return:
            np.array: Output array of shape (batch_size, output_dim).
        """
        Q = explicit.Q
        b = explicit.b
        z = x @ Q.T + b
        return z
    
    def direct_to_explicit(self):
        """
        Get explicit parameters for the Cayley linear layer.
        return:
            params: Params containing the explicit parameters.
        """
        Q = cayley((self.alpha / norm(self.weight, eps=0)) * self.weight)
        return Params(Q=Q.numpy(force=True), b=self.bias.numpy(force=True))
    
    def inverse(self, y):
        """
        Inverse of the Cayley linear layer.
        Args:
            y (torch.Tensor): Input tensor to be inverted.
        Returns:
            torch.Tensor: Inverted tensor.
        """
        orth_params = self.direct_to_explicit()
        return  (y - orth_params.b) @ orth_params.Q
