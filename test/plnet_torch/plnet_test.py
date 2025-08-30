# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

from robustnn.plnet_torch.bilipnet import BiLipNet
from robustnn.plnet_torch.plnet import PLNet
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
plnet_layer = PLNet(BiLipBlock=bilipnet_layer)

# call results: tensor([ 2.3951, 10.4903], grad_fn=<MulBackward0>)
print("call results:", plnet_layer(random_input))

# explict call results: [ 2.3950937 10.490302 ]
explict_call_res = plnet_layer.explicit_call(random_input.numpy(force=True), plnet_layer.direct_to_explicit())
print("explict call results:", explict_call_res )

random_input_min = torch.tensor([[2.0,3.0],[4.0,5.0]])
# explict call with gt:  [0.7159801 0.7159799]
print("explict call with gt: ", plnet_layer.explicit_call(random_input.numpy(force=True), 
                                                          plnet_layer.direct_to_explicit(x_optimal=random_input_min.numpy(force=True)) ))

random_input_min = random_input
# explict call with same gt:  [0. 0.]
print("explict call with same gt: ", plnet_layer.explicit_call(random_input.numpy(force=True), 
                                                          plnet_layer.direct_to_explicit(x_optimal=random_input_min.numpy(force=True)) ))

