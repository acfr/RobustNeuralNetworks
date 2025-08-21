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
        y = 0.5 * ((x - x0) ** 2).sum(dim=-1)

        if self.use_bias:
            y += self.bias
        return y 
    
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