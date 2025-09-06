# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

from robustnn.plnet_torch.monlipnet import MonLipNet
import torch
import numpy as np

# Set seeds for all RNGs
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)        # for current GPU
torch.cuda.manual_seed_all(seed)    # for all GPUs
np.random.seed(seed)

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
monlipnet_layer = MonLipNet(features=input_size, 
                            unit_features=units,
                            mu=mu,
                            nu=nu,
                            tau=tau,
                            is_mu_fixed=False,
                            is_nu_fixed=False,
                            is_tau_fixed=True, 
                            )

# Bound:  (1.0, 2.0, 2.0)
print("Bound: ", monlipnet_layer.get_bounds())

# call results: tensor([[1.2342, 3.2445],
        # [3.5813, 6.4334]], grad_fn=<AddBackward0>)
print("call results:", monlipnet_layer( random_input))

# explict call results: [[1.2342471 3.2444794]
#  [3.5812564 6.433382 ]]
explict_call_res = monlipnet_layer.explicit_call( random_input.numpy(force=True), monlipnet_layer.direct_to_explicit())
print("explict call results:",  explict_call_res)

# inverse: [[1.        2.0000005]
#  [3.        4.       ]]
print("inverse:", monlipnet_layer.inverse(explict_call_res))

                                        