# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

from robustnn.plnet_torch.bilipnet import BiLipNet
import torch
import numpy as np


# Generate random input
batches = 1
input_size = 2
inputs = torch.zeros((batches, input_size))
random_input = torch.tensor([[1.0,2.0],[3.0,4.0]])
units = [2,2]
depth = 2
mu = 1
nu = 2
tau = 2

# Initialize a unitary layer
bilipnet_layer = BiLipNet(features=input_size, 
                            unit_features=units,
                            mu=mu,
                            nu=nu,
                            tau=tau,
                            is_mu_fixed=True,
                            is_nu_fixed=True,
                            is_tau_fixed=False, 
                            depth=depth,
                            )

# Bound:  (1.0, 1.9999999315429164, 1.9999999315429164)
print("Bound: ", bilipnet_layer.get_bounds())

# call results: tensor([[2.2373, 2.9400],
#         [3.8189, 6.3574]], grad_fn=<AddmmBackward0>)
print("call results:", bilipnet_layer(random_input))

# explict call results: [[2.2372503 2.9399738]
#  [3.8188756 6.357416 ]]
explict_call_res = bilipnet_layer.explicit_call(random_input.numpy(force=True), bilipnet_layer.direct_to_explicit())
print("explict call results:", explict_call_res )

# inverse: [[0.99999946 1.9999995 ]
#  [2.9999998  3.9999983 ]]
print("inverse:", bilipnet_layer.inverse(explict_call_res,
                                         alphas=[0.1]*2,
                                         inverse_activation_fns=[lambda x: np.maximum(0, x), lambda x: np.maximum(0, x)],
                                         iterations=[500, 500],
                                         Lambdas=[1, 1] ) )